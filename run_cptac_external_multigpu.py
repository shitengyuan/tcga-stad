#!/usr/bin/env python3
"""
Distributed external inference on CPTAC-STAD SVS slides.

This script:
  1. samples tissue patches from downloaded SVS files,
  2. extracts UNI2-h 1536-d patch features,
  3. runs the saved ABMIL panel models M1-M4,
  4. writes slide-level and patient-level prediction CSV files.

Launch example:
  torchrun --standalone --nproc_per_node=4 run_cptac_external_multigpu.py \
    --svs_dir /path/to/cptac-stad-histopathology \
    --uni_weights /path/to/uni2-h-weights/pytorch_model.bin \
    --model_dir models \
    --out_dir results/external_cptac \
    --max_patches 8192 \
    --batch_size 64

Notes:
  - AUC needs gold labels. Use --labels_csv when a CPTAC label table is ready.
  - The default target_mpp=0.5 reads 20x-equivalent tissue from 40x Aperio slides.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.layers import SwiGLUPacked
from timm.models.vision_transformer import VisionTransformer
from tqdm import tqdm

try:
    import openslide
except (ImportError, OSError) as e:
    raise SystemExit(
        "Cannot load OpenSlide. Install both the Python package and native libraries.\n"
        "For this cluster, the common fix is:\n"
        "  conda install -y --override-channels "
        "-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main pcre openslide\n"
        "If conda is slow on a GPU node, run the install on a login node first.\n"
        f"Original error: {e}"
    ) from e


BASE = Path(__file__).resolve().parent
DEFAULT_SVS_DIR = BASE.parent / "dataset" / "cptac-stad-histopathology"
DEFAULT_UNI_WEIGHTS = BASE.parent / "uni2-h-weights" / "pytorch_model.bin"
DEFAULT_MODEL_DIR = BASE / "models"
DEFAULT_OUT_DIR = BASE / "results" / "external_cptac"

MODEL_FILES = {
    "M1_immune_sensitive": "M1_immune_sensitive.pt",
    "M2_msi": "M2_msi.pt",
    "M3_ebv": "M3_ebv.pt",
    "M4_subtype4": "M4_subtype4.pt",
}

SUBTYPE4_NAMES = ["EBV", "MSI", "GS", "CIN"]


class ABMILClassifier(nn.Module):
    def __init__(self, in_dim=1536, hidden=256, n_classes=2, dropout=0.25):
        super().__init__()
        self.n_classes = n_classes
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.att_V = nn.Linear(hidden, 128)
        self.att_U = nn.Linear(hidden, 128)
        self.att_w = nn.Linear(128, 1)
        self.clf = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.proj(x)
        v = torch.tanh(self.att_V(x))
        u = torch.sigmoid(self.att_U(x))
        a = F.softmax(self.att_w(v * u).squeeze(-1), dim=0)
        z = (x * a.unsqueeze(-1)).sum(0)
        return self.clf(z).unsqueeze(0), a


def distributed_setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world


def distributed_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def log0(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def patient_id_from_slide(path: Path) -> str:
    parts = path.stem.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else path.stem


def load_uni2_encoder(weights_path: Path, device: torch.device):
    model = VisionTransformer(
        img_size=224,
        patch_size=14,
        in_chans=3,
        num_classes=0,
        global_pool="token",
        embed_dim=1536,
        depth=24,
        num_heads=24,
        mlp_ratio=8192 / 1536,
        mlp_layer=SwiGLUPacked,
        qkv_bias=True,
        init_values=1e-5,
        class_token=True,
        reg_tokens=8,
        no_embed_class=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    state = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[UNI2] non-strict load: missing={len(missing)} unexpected={len(unexpected)}",
            file=sys.stderr,
            flush=True,
        )
        if missing:
            print(f"[UNI2] first missing: {missing[:8]}", file=sys.stderr, flush=True)
        if unexpected:
            print(f"[UNI2] first unexpected: {unexpected[:8]}", file=sys.stderr, flush=True)
    model.eval().to(device)
    return model


def load_abmil_panel(model_dir: Path, device: torch.device):
    panel = {}
    for name, fname in MODEL_FILES.items():
        ckpt_path = model_dir / fname
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        model = ABMILClassifier(
            in_dim=int(cfg.get("in_dim", 1536)),
            hidden=int(cfg.get("hidden", 256)),
            n_classes=int(cfg.get("n_classes", 2)),
            dropout=float(cfg.get("dropout", 0.25)),
        )
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval().to(device)
        panel[name] = model
    return panel


def tissue_mask_from_thumbnail(slide, mask_max_size: int):
    w, h = slide.dimensions
    scale = min(mask_max_size / max(w, h), 1.0)
    thumb_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    thumb = slide.get_thumbnail(thumb_size).convert("RGB")
    arr = np.asarray(thumb).astype(np.float32) / 255.0
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = maxc - minc
    # H&E tissue is usually darker and more saturated than white background.
    mask = (sat > 0.05) & (maxc < 0.92)
    return mask.astype(np.uint8), scale


def sample_tissue_coords(
    slide,
    *,
    patch_size: int,
    target_mpp: float,
    max_patches: int,
    stride_factor: float,
    tissue_threshold: float,
    mask_max_size: int,
    seed: int,
):
    mpp_raw = slide.properties.get("openslide.mpp-x") or slide.properties.get("aperio.MPP")
    native_mpp = float(mpp_raw) if mpp_raw else target_mpp
    read_size = max(1, int(round(patch_size * target_mpp / native_mpp)))
    stride = max(1, int(round(read_size * stride_factor)))

    mask, scale = tissue_mask_from_thumbnail(slide, mask_max_size)
    w, h = slide.dimensions
    coords = []

    for y in range(0, max(1, h - read_size + 1), stride):
        y0 = max(0, int(round(y * scale)))
        y1 = min(mask.shape[0], int(round((y + read_size) * scale)))
        if y1 <= y0:
            continue
        for x in range(0, max(1, w - read_size + 1), stride):
            x0 = max(0, int(round(x * scale)))
            x1 = min(mask.shape[1], int(round((x + read_size) * scale)))
            if x1 <= x0:
                continue
            if mask[y0:y1, x0:x1].mean() >= tissue_threshold:
                coords.append((x, y))

    rng = random.Random(seed)
    rng.shuffle(coords)
    if max_patches and len(coords) > max_patches:
        coords = coords[:max_patches]
    coords.sort(key=lambda xy: (xy[1], xy[0]))
    return coords, read_size, native_mpp


def preprocess_batch(images):
    arr = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in images], axis=0)
    arr = torch.from_numpy(arr).permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (arr - mean) / std


@torch.inference_mode()
def extract_features_for_slide(
    slide_path: Path,
    uni,
    device: torch.device,
    *,
    patch_size: int,
    target_mpp: float,
    max_patches: int,
    stride_factor: float,
    tissue_threshold: float,
    mask_max_size: int,
    batch_size: int,
    seed: int,
):
    slide = openslide.OpenSlide(str(slide_path))
    coords, read_size, native_mpp = sample_tissue_coords(
        slide,
        patch_size=patch_size,
        target_mpp=target_mpp,
        max_patches=max_patches,
        stride_factor=stride_factor,
        tissue_threshold=tissue_threshold,
        mask_max_size=mask_max_size,
        seed=seed,
    )
    if not coords:
        raise RuntimeError("no tissue patches sampled")

    feats = []
    use_amp = device.type == "cuda"
    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start : start + batch_size]
        imgs = []
        for x, y in batch_coords:
            img = slide.read_region((int(x), int(y)), 0, (read_size, read_size)).convert("RGB")
            if read_size != patch_size:
                img = img.resize((patch_size, patch_size), Image.Resampling.BICUBIC)
            imgs.append(img)
        x = preprocess_batch(imgs).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            z = uni(x)
        feats.append(z.float().cpu())
    slide.close()
    return torch.cat(feats, dim=0), coords, read_size, native_mpp


@torch.inference_mode()
def predict_panel(features_cpu: torch.Tensor, panel, device: torch.device):
    x = features_cpu.to(device, non_blocking=True)
    out = {}
    for name, model in panel.items():
        logits, attn = model(x)
        prob = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        out[name] = prob
        if name == "M1_immune_sensitive":
            top_idx = torch.topk(attn.detach().cpu(), k=min(20, attn.numel())).indices.numpy()
            out["_m1_top_attention_idx"] = top_idx
    return out


def row_from_prediction(slide_path, n_patches, read_size, native_mpp, pred):
    row = {
        "slide_id": slide_path.stem,
        "patient_id": patient_id_from_slide(slide_path),
        "svs_path": str(slide_path),
        "n_patches": n_patches,
        "read_size_level0": read_size,
        "native_mpp": round(float(native_mpp), 6),
    }
    for model_name in MODEL_FILES:
        probs = pred[model_name]
        for i, p in enumerate(probs):
            row[f"{model_name}_prob_c{i}"] = round(float(p), 6)
        row[f"{model_name}_pred_class"] = int(np.argmax(probs))
    row["immune_sensitive_prob"] = row["M1_immune_sensitive_prob_c1"]
    row["immune_sensitive_pred"] = int(row["immune_sensitive_prob"] >= 0.5)
    row["msi_prob"] = row["M2_msi_prob_c1"]
    row["ebv_prob"] = row["M3_ebv_prob_c1"]
    row["subtype4_pred"] = SUBTYPE4_NAMES[row["M4_subtype4_pred_class"]]
    return row


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def combine_rank_outputs(out_dir: Path, world: int):
    frames = []
    for r in range(world):
        p = out_dir / f"slide_predictions.rank{r}.csv"
        if p.exists() and p.stat().st_size > 0:
            frames.append(pd.read_csv(p))
    if not frames:
        return None, None
    slide_df = pd.concat(frames, ignore_index=True).sort_values(["patient_id", "slide_id"])
    slide_out = out_dir / "external_cptac_slide_predictions.csv"
    slide_df.to_csv(slide_out, index=False)

    prob_cols = [c for c in slide_df.columns if c.endswith("_prob_c0") or c.endswith("_prob_c1")]
    prob_cols += [c for c in slide_df.columns if c.startswith("M4_subtype4_prob_c")]
    prob_cols = sorted(set(prob_cols), key=list(slide_df.columns).index)
    agg_cols = ["patient_id", *prob_cols, "immune_sensitive_prob", "msi_prob", "ebv_prob"]
    patient_df = slide_df[agg_cols].groupby("patient_id", as_index=False).mean(numeric_only=True)
    patient_df["immune_sensitive_pred"] = (patient_df["immune_sensitive_prob"] >= 0.5).astype(int)
    m4_cols = [f"M4_subtype4_prob_c{i}" for i in range(4)]
    patient_df["subtype4_pred"] = [
        SUBTYPE4_NAMES[int(i)] for i in patient_df[m4_cols].to_numpy().argmax(axis=1)
    ]
    patient_out = out_dir / "external_cptac_patient_predictions.csv"
    patient_df.to_csv(patient_out, index=False)
    return slide_out, patient_out


def compute_metrics_if_labels(out_dir: Path, labels_csv: Path | None):
    if not labels_csv:
        return None
    from sklearn.metrics import average_precision_score, roc_auc_score

    pred = pd.read_csv(out_dir / "external_cptac_patient_predictions.csv")
    labels = pd.read_csv(labels_csv)
    if "patient_id" not in labels.columns:
        raise ValueError("--labels_csv must contain a patient_id column")
    df = pred.merge(labels, on="patient_id", how="inner")
    tasks = {
        "M1_immune_sensitive": ("immune_sensitive", "immune_sensitive_prob"),
        "M2_msi": ("msi", "msi_prob"),
        "M3_ebv": ("ebv", "ebv_prob"),
    }
    metrics = {"n_labeled_patients": int(len(df)), "tasks": {}}
    for task, (label_col, score_col) in tasks.items():
        if label_col not in df.columns:
            continue
        y = pd.to_numeric(df[label_col], errors="coerce")
        s = pd.to_numeric(df[score_col], errors="coerce")
        mask = y.notna() & s.notna()
        yv = y[mask].astype(int).to_numpy()
        sv = s[mask].to_numpy()
        if len(yv) == 0 or len(np.unique(yv)) < 2:
            metrics["tasks"][task] = {"n": int(len(yv)), "error": "need both classes for AUC"}
            continue
        metrics["tasks"][task] = {
            "n": int(len(yv)),
            "n_pos": int(yv.sum()),
            "auc": float(roc_auc_score(yv, sv)),
            "ap": float(average_precision_score(yv, sv)),
        }
    metrics_out = out_dir / "external_cptac_metrics.json"
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--svs_dir", type=Path, default=DEFAULT_SVS_DIR)
    p.add_argument("--uni_weights", type=Path, default=DEFAULT_UNI_WEIGHTS)
    p.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--pattern", default="*.svs")
    p.add_argument("--max_slides", type=int, default=None)
    p.add_argument("--max_patches", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patch_size", type=int, default=224)
    p.add_argument("--target_mpp", type=float, default=0.5)
    p.add_argument("--stride_factor", type=float, default=1.0)
    p.add_argument("--tissue_threshold", type=float, default=0.35)
    p.add_argument("--mask_max_size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labels_csv", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world = distributed_setup()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slides = sorted(args.svs_dir.glob(args.pattern))
    if args.max_slides:
        slides = slides[: args.max_slides]
    my_slides = [p for i, p in enumerate(slides) if i % world == rank]

    log0(rank, f"Found {len(slides)} slides. world_size={world}. out_dir={args.out_dir}")
    print(f"[rank {rank}] device={device} slides={len(my_slides)}", flush=True)

    uni = load_uni2_encoder(args.uni_weights, device)
    panel = load_abmil_panel(args.model_dir, device)

    rows = []
    errors = []
    iterator = tqdm(my_slides, desc=f"rank{rank}", disable=False)
    for slide_i, slide_path in enumerate(iterator):
        try:
            features, coords, read_size, native_mpp = extract_features_for_slide(
                slide_path,
                uni,
                device,
                patch_size=args.patch_size,
                target_mpp=args.target_mpp,
                max_patches=args.max_patches,
                stride_factor=args.stride_factor,
                tissue_threshold=args.tissue_threshold,
                mask_max_size=args.mask_max_size,
                batch_size=args.batch_size,
                seed=args.seed + rank * 100000 + slide_i,
            )
            pred = predict_panel(features, panel, device)
            rows.append(row_from_prediction(slide_path, len(coords), read_size, native_mpp, pred))
        except Exception as e:
            errors.append({"slide_id": slide_path.stem, "svs_path": str(slide_path), "error": repr(e)})
            print(f"[rank {rank}] ERROR {slide_path.name}: {e}", file=sys.stderr, flush=True)

    rank_csv = args.out_dir / f"slide_predictions.rank{rank}.csv"
    write_rows(rank_csv, rows)
    err_path = args.out_dir / f"errors.rank{rank}.json"
    err_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print(f"[rank {rank}] wrote {rank_csv} rows={len(rows)} errors={len(errors)}", flush=True)

    if world > 1:
        dist.barrier()
    if rank == 0:
        slide_out, patient_out = combine_rank_outputs(args.out_dir, world)
        if slide_out:
            print(f"[rank 0] combined slide predictions: {slide_out}", flush=True)
            print(f"[rank 0] combined patient predictions: {patient_out}", flush=True)
            metrics_out = compute_metrics_if_labels(args.out_dir, args.labels_csv)
            if metrics_out:
                print(f"[rank 0] metrics: {metrics_out}", flush=True)
    distributed_cleanup()


if __name__ == "__main__":
    main()
