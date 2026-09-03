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
DEFAULT_OUT = ROOT / "results" / "visual_evidence_package"
M4_CLASSES = ["EBV", "MSI", "GS", "CIN"]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ".-_" else "_" for c in value)


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
    attention = np.asarray(attention, dtype=float).reshape(-1)
    if attention.max() > attention.min():
        c = (attention - attention.min()) / (attention.max() - attention.min())
    else:
        c = attention
    fig, ax = plt.subplots(figsize=(7, 6))
    point_size = float(np.clip(50000 / max(len(coords), 1), 0.2, 5.0))
    sc = ax.scatter(coords[:, 0], -coords[:, 1], c=c, s=point_size, cmap="magma", linewidths=0, rasterized=True)
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


def normalize_all_h5_rows(feature_dir: Path) -> pd.DataFrame:
    paths = sorted(feature_dir.glob("*.h5"))
    rows = []
    for path in paths:
        slide_id = path.stem
        rows.append(
            {
                "source_table": f"feature_dir:{display_path(feature_dir)}",
                "patient_id": slide_id[:12] if slide_id.startswith("TCGA-") else slide_id.split(".")[0],
                "slide_id": slide_id,
                "h5_path": display_path(path),
                "raw_svs_paths": "",
                "case_group": "all_feature_slides",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature_dir", type=Path, default=ROOT / "tcga_stad_uni2h" / "TCGA-STAD" / "features")
    parser.add_argument("--all_feature_slides", action="store_true", help="Generate attention heatmaps for every .h5 under --feature_dir.")
    parser.add_argument("--max_patches_attention", type=int, default=12000)
    parser.add_argument(
        "--all_patches_attention",
        action="store_true",
        help="Use every patch in each h5 when drawing attention heatmaps; overrides --max_patches_attention.",
    )
    parser.add_argument("--max_thumbnail_size", type=int, default=1600)
    parser.add_argument("--representative_cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    max_patches_attention = None if args.all_patches_attention else args.max_patches_attention
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_m1(device)
    cluster_path = ROOT / "results" / "cpu_supplement" / "unsupervised_cluster_numeric_evidence" / "sampled_patch_cluster_assignments.csv"
    clusters = pd.read_csv(cluster_path) if cluster_path.exists() else pd.DataFrame()

    feature_dir = args.feature_dir if args.feature_dir.is_absolute() else ROOT / args.feature_dir
    if args.all_feature_slides:
        cases = normalize_all_h5_rows(feature_dir)
    else:
        cases = pd.concat(
            [normalize_showcase_rows(), normalize_representative_rows(args.representative_cases)],
            ignore_index=True,
        ).drop_duplicates(["patient_id", "slide_id"])

    rows: list[dict[str, Any]] = []
    for _, case in cases.iterrows():
        pid = str(case["patient_id"])
        slide_id = str(case["slide_id"])
        slide_safe = safe_name(slide_id)
        h5_value = Path(str(case["h5_path"]))
        h5_path = h5_value if h5_value.is_absolute() else ROOT / h5_value
        case_dir = out_dir / pid
        rec: dict[str, Any] = {
            "patient_id": pid,
            "slide_id": slide_id,
            "case_group": case["case_group"],
            "h5_path": display_path(h5_path) if h5_path.exists() else str(h5_path),
            "thumbnail_path": "",
            "thumbnail_status": "not_available",
            "attention_heatmap_path": "",
            "attention_status": "not_run",
            "n_patches_total": "",
            "n_patches_attention_plotted": "",
            "attention_patch_sampling": "all_patches" if args.all_patches_attention else f"max_{args.max_patches_attention}",
            "cluster_overlay_path": "",
            "cluster_status": "not_run",
        }

        raw_paths = [Path(p) for p in str(case.get("raw_svs_paths", "")).split(";") if p and p != "nan"]
        raw_paths = [ROOT / p if not p.is_absolute() else p for p in raw_paths]
        raw_existing = [p for p in raw_paths if p.exists()]
        if raw_existing:
            thumb_path = case_dir / f"{slide_safe}_thumbnail.jpg"
            rec["thumbnail_status"] = save_thumbnail(thumb_path, raw_existing[0], args.max_thumbnail_size)
            if rec["thumbnail_status"] == "generated":
                rec["thumbnail_path"] = display_path(thumb_path)

        if h5_path.exists():
            try:
                with h5py.File(h5_path, "r") as h:
                    feat_ds = h["features"]
                    rec["n_patches_total"] = int(feat_ds.shape[1] if len(feat_ds.shape) == 3 else feat_ds.shape[0])
                feat, coords = read_h5(h5_path, max_patches_attention, args.seed)
                rec["n_patches_attention_plotted"] = int(len(coords))
                with torch.no_grad():
                    x = torch.from_numpy(feat).to(device)
                    logits, att = model(x)
                    prob = torch.softmax(logits, dim=1)[0, 1].item()
                att_path = case_dir / f"{slide_safe}_M1_attention_coords.png"
                save_attention_plot(att_path, coords, att.detach().cpu().numpy(), f"{slide_id} M1 attention (prob={prob:.3f})")
                rec["attention_heatmap_path"] = display_path(att_path)
                rec["attention_status"] = "generated_coordinate_heatmap"
                rec["M1_prob_from_checkpoint"] = prob
            except Exception as exc:
                rec["attention_status"] = f"failed:{type(exc).__name__}:{exc}"

        if not clusters.empty:
            sub = clusters[clusters["slide_id"].astype(str).eq(slide_id)].copy()
            if len(sub):
                cl_path = case_dir / f"{slide_safe}_cluster_coords.png"
                save_cluster_plot(cl_path, sub, f"{slide_id} numeric cluster overlay")
                rec["cluster_overlay_path"] = display_path(cl_path)
                rec["cluster_status"] = "generated_coordinate_overlay"
            else:
                rec["cluster_status"] = "no_sampled_cluster_rows_for_slide"
        rows.append(rec)

    out_csv = out_dir / "visual_evidence_manifest.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "out_dir": display_path(out_dir),
        "n_cases": int(len(rows)),
        "case_source": "all_feature_slides" if args.all_feature_slides else "showcase_plus_representative_cases",
        "feature_dir": display_path(feature_dir),
        "attention_patch_mode": "all_patches" if args.all_patches_attention else f"max_{args.max_patches_attention}",
        "total_patches_visualized": int(sum(int(r["n_patches_attention_plotted"] or 0) for r in rows)),
        "thumbnail_generated": int(sum(r["thumbnail_status"] == "generated" for r in rows)),
        "attention_generated": int(sum(r["attention_status"] == "generated_coordinate_heatmap" for r in rows)),
        "cluster_generated": int(sum(r["cluster_status"] == "generated_coordinate_overlay" for r in rows)),
        "important_limitation": "Attention and cluster figures are coordinate plots. Pathology naming of clusters still requires rendered representative patches and pathologist review.",
    }
    (out_dir / "visual_evidence_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
