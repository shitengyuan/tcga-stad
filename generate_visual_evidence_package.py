#!/usr/bin/env python3
"""Generate lightweight visual evidence from existing h5 coords/features and available SVS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.train_multitask import ABMILClassifier


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "visual_evidence_package"
M4_CLASSES = ["EBV", "MSI", "GS", "CIN"]


def read_h5(path: Path, max_patches: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h:
        feat_ds = h["features"]
        coords_ds = h["coords_patching"]
        n = feat_ds.shape[1] if len(feat_ds.shape) == 3 else feat_ds.shape[0]
        if max_patches and n > max_patches:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, max_patches, replace=False)
            idx.sort()
            feat = feat_ds[0, idx, :] if len(feat_ds.shape) == 3 else feat_ds[idx, :]
            coords = coords_ds[idx, :]
        else:
            feat = feat_ds[0] if len(feat_ds.shape) == 3 else feat_ds[:]
            coords = coords_ds[:]
    return feat.astype(np.float32), coords.astype(np.int64)


def load_m1(device: torch.device) -> ABMILClassifier:
    ckpt = torch.load(ROOT / "models" / "M1_immune_sensitive.pt", map_location=device)
    cfg = ckpt.get("config", {"in_dim": 1536, "hidden": 256, "n_classes": 2, "dropout": 0.25})
    model = ABMILClassifier(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def save_attention_plot(path: Path, coords: np.ndarray, attention: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    attention = np.asarray(attention, dtype=float)
    if attention.max() > attention.min():
        c = (attention - attention.min()) / (attention.max() - attention.min())
    else:
        c = attention
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], -coords[:, 1], c=c, s=5, cmap="magma", linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("x(level-0)")
    ax.set_ylabel("-y(level-0)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="normalized attention")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_cluster_plot(path: Path, rows: pd.DataFrame, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(rows["coord_x"], -rows["coord_y"], c=rows["cluster"], s=8, cmap="tab10", linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("x(level-0)")
    ax.set_ylabel("-y(level-0)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label="numeric cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_thumbnail(path: Path, svs_path: Path, max_size: int) -> str:
    try:
        import openslide

        slide = openslide.OpenSlide(str(svs_path))
        thumb = slide.get_thumbnail((max_size, max_size)).convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(path, quality=90)
        slide.close()
        return "generated"
    except Exception as exc:
        return f"failed:{type(exc).__name__}:{exc}"


def normalize_showcase_rows() -> pd.DataFrame:
    path = ROOT / "results" / "cpu_supplement" / "showcase_cases_evidence_with_h5_svs.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "source_table": str(path.relative_to(ROOT)),
            "patient_id": df["patient_id"],
            "slide_id": df["h5_slide_id"],
            "h5_path": df["h5_path"],
            "raw_svs_paths": df.get("raw_svs_paths", ""),
            "case_group": "showcase",
        }
    )
    return out


def normalize_representative_rows(max_cases: int) -> pd.DataFrame:
    path = ROOT / "results" / "cpu_supplement" / "representative_case_candidate_package_20.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path).head(max_cases)
    out = pd.DataFrame(
        {
            "source_table": str(path.relative_to(ROOT)),
            "patient_id": df["patient_id"],
            "slide_id": df["slide_id"],
            "h5_path": df["path"],
            "raw_svs_paths": "",
            "case_group": "representative_" + df["case_group"].astype(str),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_patches_attention", type=int, default=12000)
    parser.add_argument("--max_thumbnail_size", type=int, default=1600)
    parser.add_argument("--representative_cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_m1(device)
    cluster_path = ROOT / "results" / "cpu_supplement" / "unsupervised_cluster_numeric_evidence" / "sampled_patch_cluster_assignments.csv"
    clusters = pd.read_csv(cluster_path) if cluster_path.exists() else pd.DataFrame()

    cases = pd.concat(
        [normalize_showcase_rows(), normalize_representative_rows(args.representative_cases)],
        ignore_index=True,
    ).drop_duplicates(["patient_id", "slide_id"])

    rows: list[dict[str, Any]] = []
    for _, case in cases.iterrows():
        pid = str(case["patient_id"])
        slide_id = str(case["slide_id"])
        h5_path = ROOT / str(case["h5_path"])
        case_dir = OUT / pid
        rec: dict[str, Any] = {
            "patient_id": pid,
            "slide_id": slide_id,
            "case_group": case["case_group"],
            "h5_path": str(h5_path.relative_to(ROOT)) if h5_path.exists() else str(h5_path),
            "thumbnail_path": "",
            "thumbnail_status": "not_available",
            "attention_heatmap_path": "",
            "attention_status": "not_run",
            "cluster_overlay_path": "",
            "cluster_status": "not_run",
        }

        raw_paths = [Path(p) for p in str(case.get("raw_svs_paths", "")).split(";") if p and p != "nan"]
        raw_paths = [ROOT / p if not p.is_absolute() else p for p in raw_paths]
        raw_existing = [p for p in raw_paths if p.exists()]
        if raw_existing:
            thumb_path = case_dir / f"{pid}_thumbnail.jpg"
            rec["thumbnail_status"] = save_thumbnail(thumb_path, raw_existing[0], args.max_thumbnail_size)
            if rec["thumbnail_status"] == "generated":
                rec["thumbnail_path"] = str(thumb_path.relative_to(ROOT))

        if h5_path.exists():
            try:
                feat, coords = read_h5(h5_path, args.max_patches_attention, args.seed)
                with torch.no_grad():
                    x = torch.from_numpy(feat).to(device)
                    logits, att = model(x)
                    prob = torch.softmax(logits, dim=1)[0, 1].item()
                att_path = case_dir / f"{pid}_M1_attention_coords.png"
                save_attention_plot(att_path, coords, att.detach().cpu().numpy(), f"{pid} M1 attention (prob={prob:.3f})")
                rec["attention_heatmap_path"] = str(att_path.relative_to(ROOT))
                rec["attention_status"] = "generated_coordinate_heatmap"
                rec["M1_prob_from_checkpoint"] = prob
            except Exception as exc:
                rec["attention_status"] = f"failed:{type(exc).__name__}:{exc}"

        if not clusters.empty:
            sub = clusters[clusters["slide_id"].astype(str).eq(slide_id)].copy()
            if len(sub):
                cl_path = case_dir / f"{pid}_cluster_coords.png"
                save_cluster_plot(cl_path, sub, f"{pid} numeric cluster overlay")
                rec["cluster_overlay_path"] = str(cl_path.relative_to(ROOT))
                rec["cluster_status"] = "generated_coordinate_overlay"
            else:
                rec["cluster_status"] = "no_sampled_cluster_rows_for_slide"
        rows.append(rec)

    out_csv = OUT / "visual_evidence_manifest.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "out_dir": str(OUT.relative_to(ROOT)),
        "n_cases": int(len(rows)),
        "thumbnail_generated": int(sum(r["thumbnail_status"] == "generated" for r in rows)),
        "attention_generated": int(sum(r["attention_status"] == "generated_coordinate_heatmap" for r in rows)),
        "cluster_generated": int(sum(r["cluster_status"] == "generated_coordinate_overlay" for r in rows)),
        "important_limitation": "Attention and cluster figures are coordinate plots. Pathology naming of clusters still requires rendered representative patches and pathologist review.",
    }
    (OUT / "visual_evidence_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
