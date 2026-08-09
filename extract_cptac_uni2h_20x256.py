#!/usr/bin/env python3
"""
Extract UNI2-h features from CPTAC-STAD SVS slides using the same sampling
convention as MahmoodLab UNI2-h-features:

  - tissue patches at 20x-equivalent magnification (target_mpp=0.5)
  - 256 x 256 pixel pathology patches
  - UNI2-h encoder input resized to 224 x 224, matching the ViT checkpoint
  - one output feature file per slide

Output .pt format:
  {
    "features":        FloatTensor[N, 1536],
    "coords_patching": LongTensor[N, 2],   # level-0 x/y coordinates
    "metadata":        {...}
  }

Output .h5 format:
  features:        float32[1, N, 1536]
  coords_patching: int64[N, 2]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

try:
    import h5py
except ImportError:
    h5py = None

from run_cptac_external_multigpu import (
    DEFAULT_SVS_DIR,
    DEFAULT_UNI_WEIGHTS,
    distributed_cleanup,
    distributed_setup,
    load_uni2_encoder,
    log0,
    patient_id_from_slide,
    preprocess_batch,
    sample_tissue_coords,
)

BASE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = BASE / "results" / "external_cptac_features_20x256"


def is_complete_feature_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        if path.suffix.lower() in {".pt", ".pth"}:
            obj = torch.load(path, map_location="cpu")
            feat = obj["features"] if isinstance(obj, dict) else obj
            return len(feat.shape) == 2 and feat.shape[1] == 1536 and feat.shape[0] > 0
        if path.suffix.lower() in {".h5", ".hdf5"}:
            if h5py is None:
                return False
            with h5py.File(path, "r") as h:
                feat = h["features"]
                if len(feat.shape) == 3:
                    return feat.shape[0] == 1 and feat.shape[2] == 1536 and feat.shape[1] > 0
                return len(feat.shape) == 2 and feat.shape[1] == 1536 and feat.shape[0] > 0
    except Exception:
        return False
    return False


def save_feature_file(
    out_path: Path,
    *,
    slide_path: Path,
    features: torch.Tensor,
    coords: list[tuple[int, int]],
    read_size_level0: int,
    native_mpp: float,
    target_mpp: float,
    patch_size_20x: int,
    encoder_input_size: int,
    compression: str | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()

    feat_cpu = features.detach().cpu().float().contiguous()
    coord_np = np.asarray(coords, dtype=np.int64)
    meta = {
        "slide_id": slide_path.stem,
        "patient_id": patient_id_from_slide(slide_path),
        "svs_path": str(slide_path),
        "n_patches": int(feat_cpu.shape[0]),
        "feature_dim": int(feat_cpu.shape[1]),
        "target_mpp": float(target_mpp),
        "native_mpp": float(native_mpp),
        "patch_size_20x": int(patch_size_20x),
        "encoder_input_size": int(encoder_input_size),
        "read_size_level0": int(read_size_level0),
        "encoder": "UNI2-h",
    }

    if out_path.suffix.lower() in {".pt", ".pth"}:
        torch.save(
            {
                "features": feat_cpu,
                "coords_patching": torch.from_numpy(coord_np),
                "metadata": meta,
            },
            tmp_path,
        )
    elif out_path.suffix.lower() in {".h5", ".hdf5"}:
        if h5py is None:
            raise RuntimeError("h5py is required for --format h5")
        feat_np = feat_cpu.numpy().astype(np.float32, copy=False)
        with h5py.File(tmp_path, "w") as h:
            h.create_dataset("features", data=feat_np[None, :, :], compression=compression)
            h.create_dataset("coords_patching", data=coord_np, compression=compression)
            for key, value in meta.items():
                h.attrs[key] = value
    else:
        raise ValueError(f"unsupported output suffix: {out_path.suffix}")
    tmp_path.replace(out_path)


@torch.inference_mode()
def extract_features_for_slide_20x256(
    slide_path: Path,
    uni,
    device: torch.device,
    *,
    patch_size_20x: int,
    encoder_input_size: int,
    target_mpp: float,
    max_patches: int,
    stride_factor: float,
    tissue_threshold: float,
    mask_max_size: int,
    batch_size: int,
    seed: int,
):
    import openslide

    slide = openslide.OpenSlide(str(slide_path))
    coords, read_size, native_mpp = sample_tissue_coords(
        slide,
        patch_size=patch_size_20x,
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
    resample = Image.Resampling.BICUBIC
    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start : start + batch_size]
        imgs = []
        for x, y in batch_coords:
            img = slide.read_region((int(x), int(y)), 0, (read_size, read_size)).convert("RGB")
            if read_size != patch_size_20x:
                img = img.resize((patch_size_20x, patch_size_20x), resample)
            if encoder_input_size != patch_size_20x:
                img = img.resize((encoder_input_size, encoder_input_size), resample)
            imgs.append(img)
        x = preprocess_batch(imgs).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            z = uni(x)
        feats.append(z.float().cpu())
    slide.close()
    return torch.cat(feats, dim=0), coords, read_size, native_mpp


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import csv

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def combine_rank_manifests(out_dir: Path, world: int):
    import pandas as pd

    frames = []
    for rank in range(world):
        path = out_dir / f"feature_manifest.rank{rank}.csv"
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).sort_values(["patient_id", "slide_id"])
    out = out_dir / "feature_manifest.csv"
    df.to_csv(out, index=False)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--svs_dir", type=Path, default=DEFAULT_SVS_DIR)
    p.add_argument("--uni_weights", type=Path, default=DEFAULT_UNI_WEIGHTS)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--pattern", default="*.svs")
    p.add_argument("--max_slides", type=int, default=None)
    p.add_argument("--max_patches", type=int, default=0, help="0 means all sampled tissue patches.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patch_size_20x", type=int, default=256)
    p.add_argument("--encoder_input_size", type=int, default=224)
    p.add_argument("--target_mpp", type=float, default=0.5)
    p.add_argument("--stride_factor", type=float, default=1.0)
    p.add_argument("--tissue_threshold", type=float, default=0.35)
    p.add_argument("--mask_max_size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--format", choices=["pt", "h5"], default="pt")
    p.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    p.add_argument("--overwrite", action="store_true")
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
    max_patches = None if args.max_patches == 0 else args.max_patches
    compression = None if args.compression == "none" else args.compression

    log0(rank, f"Found {len(slides)} slides. world_size={world}. out_dir={args.out_dir}")
    print(f"[rank {rank}] device={device} slides={len(my_slides)}", flush=True)

    uni = load_uni2_encoder(args.uni_weights, device)
    rows = []
    errors = []

    for slide_i, slide_path in enumerate(tqdm(my_slides, desc=f"rank{rank}")):
        out_path = args.out_dir / f"{slide_path.stem}.{args.format}"
        if not args.overwrite and is_complete_feature_file(out_path):
            rows.append(
                {
                    "slide_id": slide_path.stem,
                    "patient_id": patient_id_from_slide(slide_path),
                    "svs_path": str(slide_path),
                    "feature_path": str(out_path),
                    "n_patches": "",
                    "status": "skipped_existing",
                }
            )
            continue

        try:
            features, coords, read_size, native_mpp = extract_features_for_slide_20x256(
                slide_path,
                uni,
                device,
                patch_size_20x=args.patch_size_20x,
                encoder_input_size=args.encoder_input_size,
                target_mpp=args.target_mpp,
                max_patches=max_patches,
                stride_factor=args.stride_factor,
                tissue_threshold=args.tissue_threshold,
                mask_max_size=args.mask_max_size,
                batch_size=args.batch_size,
                seed=args.seed + rank * 100000 + slide_i,
            )
            save_feature_file(
                out_path,
                slide_path=slide_path,
                features=features,
                coords=coords,
                read_size_level0=read_size,
                native_mpp=native_mpp,
                target_mpp=args.target_mpp,
                patch_size_20x=args.patch_size_20x,
                encoder_input_size=args.encoder_input_size,
                compression=compression,
            )
            rows.append(
                {
                    "slide_id": slide_path.stem,
                    "patient_id": patient_id_from_slide(slide_path),
                    "svs_path": str(slide_path),
                    "feature_path": str(out_path),
                    "n_patches": int(features.shape[0]),
                    "status": "written",
                }
            )
        except Exception as e:
            errors.append({"slide_id": slide_path.stem, "svs_path": str(slide_path), "error": repr(e)})
            print(f"[rank {rank}] ERROR {slide_path.name}: {e}", flush=True)

    manifest_path = args.out_dir / f"feature_manifest.rank{rank}.csv"
    write_csv(manifest_path, rows)
    err_path = args.out_dir / f"errors.rank{rank}.json"
    err_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    print(f"[rank {rank}] wrote manifest={manifest_path} rows={len(rows)} errors={len(errors)}", flush=True)

    if world > 1:
        dist.barrier()
    if rank == 0:
        combined = combine_rank_manifests(args.out_dir, world)
        if combined:
            print(f"[rank 0] combined feature manifest: {combined}", flush=True)
    distributed_cleanup()


if __name__ == "__main__":
    main()
