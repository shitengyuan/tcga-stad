#!/usr/bin/env python3
"""Generate original-vs-attention-overlap panels for all slides with available SVS."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from src.train_multitask import ABMILClassifier


ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURE_MANIFEST = ROOT / "results" / "visual_evidence_package_all_feature_slides_all_patches" / "visual_evidence_manifest.csv"
DEFAULT_FEATURE_DIR = ROOT / "tcga_stad_uni2h" / "TCGA-STAD" / "features"
DEFAULT_OUT = ROOT / "results" / "visual_evidence_attention_overlap_all_available"
DEFAULT_LABEL_FILES = [
    ROOT.parent / "dataset" / "cptac-stad-histopathology" / "labels" / "cptac_stad_2026_tcga_subtype_labels_qc_pass.csv",
    ROOT / "results" / "cpu_supplement" / "showcase_cases_evidence_with_h5_svs.csv",
    ROOT / "results" / "visual_evidence_review" / "simple_gt_original_vs_attention_overlay_manifest.csv",
    ROOT / "results" / "visual_evidence_review" / "showcase_gt_original_attention_overlay_manifest.csv",
    ROOT / "results" / "locked_release_20260902" / "patient_manifest.csv",
    ROOT / "results" / "audit_first_stage" / "tcga_label_table_from_cbioportal_clinical_with_gdc_slides_and_features_after_gpu_run.csv",
    ROOT / "results" / "audit_first_stage" / "tcga_public_feature_matched_246_cohort_after_gpu_run.csv",
]
DEFAULT_SVS_DIRS = [
    ROOT / "external_downloads" / "tcga_stad" / "locked_246_svs",
    ROOT / "external_downloads" / "tcga_stad" / "showcase_primary_tumor_svs",
    ROOT.parent / "dataset" / "cptac-stad-histopathology",
]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ".-_" else "_" for c in value)


def infer_patient_id(slide_id: str) -> str:
    if slide_id.startswith("TCGA-"):
        return slide_id[:12]
    parts = slide_id.split("-")
    if len(parts) >= 2 and parts[0] in {"C3L", "C3N"}:
        return "-".join(parts[:2])
    return slide_id.split(".")[0]


def load_m1(device: torch.device) -> ABMILClassifier:
    ckpt = torch.load(ROOT / "models" / "M1_immune_sensitive.pt", map_location=device)
    cfg = ckpt.get("config", {"in_dim": 1536, "hidden": 256, "n_classes": 2, "dropout": 0.25})
    model = ABMILClassifier(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def read_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h:
        feats = h["features"]
        coords = h["coords_patching"][:] if "coords_patching" in h else h["coords"][:]
        if coords.ndim == 3:
            coords = coords[0]
        arr = feats[0] if feats.ndim == 3 else feats[:]
    return np.asarray(arr, dtype=np.float32), np.asarray(coords, dtype=np.int64)


def read_pt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        feats = obj.get("features")
        coords = obj.get("coords_patching", obj.get("coords"))
        if feats is None or coords is None:
            raise KeyError(f"{path} must contain features and coords_patching/coords")
    else:
        raise TypeError(f"Unsupported pt payload type: {type(obj).__name__}")
    if hasattr(feats, "detach"):
        feats = feats.detach().cpu().numpy()
    if hasattr(coords, "detach"):
        coords = coords.detach().cpu().numpy()
    return np.asarray(feats, dtype=np.float32), np.asarray(coords, dtype=np.int64)


def read_feature(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return read_h5(path)
    if suffix == ".pt":
        return read_pt(path)
    raise ValueError(f"Unsupported feature file suffix: {path.suffix}")


def build_cases(args: argparse.Namespace) -> pd.DataFrame:
    if args.feature_manifest.exists():
        df = pd.read_csv(args.feature_manifest)
        if args.limit:
            df = df.head(args.limit)
        feature_col = "h5_path" if "h5_path" in df.columns else "feature_path"
        if feature_col not in df.columns:
            raise KeyError(f"{args.feature_manifest} must contain h5_path or feature_path")
        out = df[["patient_id", "slide_id", feature_col]].drop_duplicates().copy()
        return out.rename(columns={feature_col: "feature_path"})
    paths = sorted(args.feature_dir.glob("*.h5")) + sorted(args.feature_dir.glob("*.pt"))
    if args.limit:
        paths = paths[: args.limit]
    return pd.DataFrame(
        {
            "patient_id": [infer_patient_id(p.stem) for p in paths],
            "slide_id": [p.stem for p in paths],
            "feature_path": [display_path(p) for p in paths],
        }
    )


def explode_slide_values(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip() and x.strip().lower() != "nan"]


def build_label_map(label_files: list[Path]) -> dict[str, dict[str, Any]]:
    label_map: dict[str, dict[str, Any]] = {}
    wanted = [
        "patient_id",
        "MSI",
        "EBV",
        "M1_label",
        "M4_subtype",
        "M4_label",
        "subtype",
        "gt_subtype",
        "TCGA_four_subtype",
        "subtype4",
        "immune_sensitive",
        "msi",
        "ebv",
    ]
    slide_cols = ["slide_id", "h5_slide_id", "all_matched_slide_ids", "gdc_primary_tumor_slide_ids", "gdc_diagnostic_slide_ids"]
    for path in label_files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            rec = {c: row.get(c, "") for c in wanted if c in df.columns}
            patient_id = str(row.get("patient_id", ""))
            if patient_id:
                label_map.setdefault(patient_id, rec)
            slide_ids: list[str] = []
            for c in slide_cols:
                if c in df.columns:
                    slide_ids.extend(explode_slide_values(row.get(c)))
            for slide_id in slide_ids:
                label_map.setdefault(slide_id, rec)
    return label_map


def label_text(patient_id: str, slide_id: str, label_map: dict[str, dict[str, Any]]) -> str:
    rec = label_map.get(slide_id) or label_map.get(patient_id) or {}
    subtype = rec.get("M4_subtype") or rec.get("TCGA_four_subtype") or rec.get("gt_subtype") or rec.get("subtype4") or rec.get("subtype") or "NA"
    msi = rec.get("MSI", rec.get("msi", "NA"))
    ebv = rec.get("EBV", rec.get("ebv", "NA"))
    m1 = rec.get("M1_label", rec.get("immune_sensitive", "NA"))
    return f"GT: {subtype} | MSI={msi} EBV={ebv} M1={m1}"


def find_svs(slide_id: str, svs_dirs: list[Path]) -> Path | None:
    candidates = [f"{slide_id}.svs", f"{slide_id}.tif", f"{slide_id}.tiff"]
    for d in svs_dirs:
        for name in candidates:
            p = d / name
            if p.exists() and p.stat().st_size > 0:
                return p
    for d in svs_dirs:
        hits = []
        for suffix in [".svs", ".tif", ".tiff"]:
            hits.extend(d.glob(f"{slide_id}*{suffix}"))
        hits = [p for p in hits if p.exists() and p.stat().st_size > 0]
        if hits:
            return sorted(hits)[0]
    return None


def normalize_attention(attention: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    a = np.asarray(attention, dtype=np.float32).reshape(-1)
    if len(a) == 0:
        return a
    lo, hi = np.percentile(a, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def make_overlay(
    thumbnail: Image.Image,
    slide_dims: tuple[int, int],
    coords: np.ndarray,
    attention: np.ndarray,
    patch_size: int,
    alpha: float,
    alpha_floor: float,
    cmap_name: str,
) -> Image.Image:
    thumb = thumbnail.convert("RGB")
    w, h = thumb.size
    sx = w / max(slide_dims[0], 1)
    sy = h / max(slide_dims[1], 1)
    heat = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    p_w = max(1, int(round(patch_size * sx)))
    p_h = max(1, int(round(patch_size * sy)))
    for (x, y), a in zip(coords, attention):
        x0 = int(round(int(x) * sx))
        y0 = int(round(int(y) * sy))
        x1 = min(w, max(0, x0) + p_w)
        y1 = min(h, max(0, y0) + p_h)
        x0 = max(0, min(w, x0))
        y0 = max(0, min(h, y0))
        if x1 <= x0 or y1 <= y0:
            continue
        heat[y0:y1, x0:x1] += float(a)
        count[y0:y1, x0:x1] += 1.0
    mask = count > 0
    heat[mask] = heat[mask] / count[mask]
    rgba = matplotlib.colormaps.get_cmap(cmap_name)(heat)
    color = (rgba[:, :, :3] * 255).astype(np.uint8)
    base = np.asarray(thumb, dtype=np.float32)
    out = base.copy()
    local_alpha = np.zeros_like(heat, dtype=np.float32)
    local_alpha[mask] = alpha_floor + (alpha - alpha_floor) * np.clip(heat[mask], 0, 1)
    local_alpha = local_alpha[:, :, None]
    out[mask] = base[mask] * (1.0 - local_alpha[mask]) + color[mask] * local_alpha[mask]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def draw_label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(16, int(min(out.size) * 0.022)))
    except Exception:
        font = ImageFont.load_default()
    margin = max(12, int(min(out.size) * 0.015))
    pad_x = max(10, int(min(out.size) * 0.012))
    pad_y = max(8, int(min(out.size) * 0.009))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x0 = max(margin, out.size[0] - tw - 2 * pad_x - margin)
    y0 = max(margin, out.size[1] - th - 2 * pad_y - margin)
    x1 = out.size[0] - margin
    y1 = out.size[1] - margin
    draw.rounded_rectangle((x0, y0, x1, y1), radius=6, fill=(0, 0, 0), outline=(255, 255, 255), width=1)
    draw.text((x0 + pad_x, y0 + pad_y), text, fill=(255, 255, 255), font=font)
    return out


def hconcat(left: Image.Image, right: Image.Image) -> Image.Image:
    h = max(left.height, right.height)
    if left.height != h:
        left = left.resize((int(left.width * h / left.height), h), Image.Resampling.LANCZOS)
    if right.height != h:
        right = right.resize((int(right.width * h / right.height), h), Image.Resampling.LANCZOS)
    gutter = max(8, int(h * 0.006))
    out = Image.new("RGB", (left.width + gutter + right.width, h), (255, 255, 255))
    out.paste(left, (0, 0))
    out.paste(right, (left.width + gutter, 0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--feature_dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--svs_dir", action="append", type=Path, default=[])
    parser.add_argument("--label_file", action="append", type=Path, default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_thumbnail_size", type=int, default=1800)
    parser.add_argument("--patch_size", type=int, default=512, help="Level-0 patch size for 20x256 features on 40x scans.")
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--alpha_floor", type=float, default=0.22, help="Minimum opacity for tissue patches after thresholding.")
    parser.add_argument("--cmap", default="turbo", help="High-contrast matplotlib colormap for H&E overlay.")
    parser.add_argument("--attention_low_pct", type=float, default=50.0)
    parser.add_argument("--attention_high_pct", type=float, default=99.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_rank < 0 or args.shard_rank >= args.num_shards:
        raise ValueError("--shard_rank must satisfy 0 <= shard_rank < num_shards")

    import openslide

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.feature_dir if args.feature_dir.is_absolute() else ROOT / args.feature_dir
    svs_dirs = args.svs_dir or DEFAULT_SVS_DIRS
    svs_dirs = [p if p.is_absolute() else ROOT / p for p in svs_dirs]
    label_files = args.label_file or DEFAULT_LABEL_FILES
    label_files = [p if p.is_absolute() else ROOT / p for p in label_files]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_m1(device)
    cases = build_cases(args)
    if args.num_shards > 1:
        cases = cases.iloc[args.shard_rank :: args.num_shards].reset_index(drop=True)
    labels = build_label_map(label_files)
    rows: list[dict[str, Any]] = []
    manifest_name = "attention_overlap_manifest.csv" if args.num_shards == 1 else f"attention_overlap_manifest.rank{args.shard_rank}.csv"
    partial_name = (
        "attention_overlap_manifest.partial.csv"
        if args.num_shards == 1
        else f"attention_overlap_manifest.rank{args.shard_rank}.partial.csv"
    )
    missing_name = (
        "attention_overlap_missing_or_failed.csv"
        if args.num_shards == 1
        else f"attention_overlap_missing_or_failed.rank{args.shard_rank}.csv"
    )
    summary_name = "attention_overlap_summary.json" if args.num_shards == 1 else f"attention_overlap_summary.rank{args.shard_rank}.json"
    index_name = "index.html" if args.num_shards == 1 else f"index.rank{args.shard_rank}.html"

    for i, case in enumerate(cases.itertuples(), 1):
        patient_id = str(case.patient_id)
        slide_id = str(case.slide_id)
        slide_safe = safe_name(slide_id)
        feature_value = Path(str(case.feature_path))
        feature_path = feature_value if feature_value.is_absolute() else ROOT / feature_value
        if not feature_path.exists():
            feature_path = feature_dir / f"{slide_id}.h5"
        if not feature_path.exists():
            feature_path = feature_dir / f"{slide_id}.pt"
        svs_path = find_svs(slide_id, svs_dirs)
        panel_path = out_dir / patient_id / f"{slide_safe}_gt_original_vs_attention_overlap.jpg"
        rec: dict[str, Any] = {
            "patient_id": patient_id,
            "slide_id": slide_id,
            "feature_path": display_path(feature_path),
            "svs_path": display_path(svs_path) if svs_path else "",
            "panel_path": display_path(panel_path),
            "status": "",
            "n_patches": "",
            "M1_prob": "",
            "gt_label_text": label_text(patient_id, slide_id, labels),
            "error": "",
        }
        if panel_path.exists() and not args.overwrite:
            rec["status"] = "skipped_existing"
            rows.append(rec)
            continue
        if svs_path is None:
            rec["status"] = "missing_svs"
            rows.append(rec)
            continue
        if not feature_path.exists():
            rec["status"] = "missing_feature"
            rows.append(rec)
            continue
        try:
            feats, coords = read_feature(feature_path)
            with torch.inference_mode():
                logits, att = model(torch.from_numpy(feats).to(device))
                prob = float(torch.softmax(logits, dim=1)[0, 1].detach().cpu().item())
            att_norm = normalize_attention(att.detach().cpu().numpy(), args.attention_low_pct, args.attention_high_pct)
            slide = openslide.OpenSlide(str(svs_path))
            thumbnail = slide.get_thumbnail((args.max_thumbnail_size, args.max_thumbnail_size)).convert("RGB")
            slide_dims = tuple(map(int, slide.dimensions))
            slide.close()
            overlay = make_overlay(thumbnail, slide_dims, coords, att_norm, args.patch_size, args.alpha, args.alpha_floor, args.cmap)
            label = rec["gt_label_text"]
            left = draw_label(thumbnail, label)
            right = draw_label(overlay, label)
            panel = hconcat(left, right)
            panel_path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(panel_path, quality=92)
            rec["status"] = "generated"
            rec["n_patches"] = int(len(coords))
            rec["M1_prob"] = prob
        except Exception as exc:
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(rec)
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(out_dir / partial_name, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / manifest_name, index=False)
    missing = df[df["status"].isin(["missing_svs", "missing_h5", "missing_feature", "failed"])].copy()
    missing.to_csv(out_dir / missing_name, index=False)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": display_path(out_dir),
        "feature_manifest": display_path(args.feature_manifest),
        "feature_dir": display_path(feature_dir),
        "svs_dirs": [display_path(p) for p in svs_dirs],
        "label_files": [display_path(p) for p in label_files],
        "num_shards": args.num_shards,
        "shard_rank": args.shard_rank,
        "status_counts": df["status"].value_counts(dropna=False).to_dict(),
        "n_rows": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()) if len(df) else 0,
        "attention_normalization": f"percentile_{args.attention_low_pct}_to_{args.attention_high_pct}",
        "attention_colormap": args.cmap,
        "attention_alpha": args.alpha,
        "attention_alpha_floor": args.alpha_floor,
        "layout": "left original WSI thumbnail, right attention-overlap WSI thumbnail, GT label on lower-right of both panels",
    }
    (out_dir / summary_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # Keep a simple CSV-driven HTML index at the root; per-patient files remain in subdirectories.
    links = []
    for row in df[df["status"].isin(["generated", "skipped_existing"])].itertuples():
        rel = Path(row.panel_path).as_posix()
        if rel.startswith(display_path(out_dir) + "/"):
            rel = rel[len(display_path(out_dir)) + 1 :]
        links.append(f"<h3>{row.patient_id} / {row.slide_id}</h3><img src='{rel}' style='max-width:100%;'>")
    (out_dir / index_name).write_text("<html><body><h1>GT Original vs Attention Overlap</h1>" + "\n".join(links) + "</body></html>", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
