#!/usr/bin/env python3
"""
Extract and save UNI2-h patch features from CPTAC-STAD SVS slides.

Output format, one feature file per slide:
  default .pt dict:
    features:        (N, 1536) float32
    coords_patching: (N, 2) int64, level-0 x/y coordinates
    metadata:        slide/patient/patch metadata
  optional .h5 with the same features and coords_patching keys.

Launch example:
  torchrun --standalone --nproc_per_node=4 extract_cptac_features_multigpu.py \
    --svs_dir /path/to/cptac-stad-histopathology \
    --uni_weights /path/to/uni2-h-weights/pytorch_model.bin \
    --out_dir results/external_cptac_features \
    --format pt \
    --max_patches 8192 \
    --batch_size 64
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
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
    extract_features_for_slide,
    load_uni2_encoder,
    log0,
    patient_id_from_slide,
)


BASE = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = BASE / "results" / "external_cptac_features"


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
                if "features" not in h:
                    return False
                feat = h["features"]
                return len(feat.shape) == 2 and feat.shape[1] == 1536 and feat.shape[0] > 0
        return False
    except Exception:
        return False


def feature_metadata(
    *,
    slide_path: Path,
    n_patches: int,
    feature_dim: int,
    read_size: int,
    native_mpp: float,
    patch_size: int,
    target_mpp: float,
) -> dict:
    return {
        "slide_id": slide_path.stem,
        "patient_id": patient_id_from_slide(slide_path),
        "svs_path": str(slide_path),
        "n_patches": int(n_patches),
        "feature_dim": int(feature_dim),
        "patch_size": int(patch_size),
        "target_mpp": float(target_mpp),
        "native_mpp": float(native_mpp),
        "read_size_level0": int(read_size),
        "encoder": "UNI2-h",
    }


def save_feature_file(
    out_path: Path,
    *,
    slide_path: Path,
    features: torch.Tensor,
    coords: list[tuple[int, int]],
    read_size: int,
    native_mpp: float,
    patch_size: int,
    target_mpp: float,
    compression: str | None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()

    feat_cpu = features.detach().cpu().float().contiguous()
    coord_np = np.asarray(coords, dtype=np.int64)
    meta = feature_metadata(
        slide_path=slide_path,
        n_patches=int(feat_cpu.shape[0]),
        feature_dim=int(feat_cpu.shape[1]),
        read_size=read_size,
        native_mpp=native_mpp,
        patch_size=patch_size,
        target_mpp=target_mpp,
    )

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
            h.create_dataset("features", data=feat_np, compression=compression)
            h.create_dataset("coords_patching", data=coord_np, compression=compression)
            for key, value in meta.items():
                h.attrs[key] = value
    else:
        raise ValueError(f"unsupported output suffix: {out_path.suffix}")
    tmp_path.replace(out_path)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import csv

    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def combine_rank_manifests(out_dir: Path, world: int):
    import pandas as pd

    frames = []
    for rank in range(world):
        path = out_dir / f"feature_manifest.rank{rank}.csv"
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    if frames:
        df = pd.concat(frames, ignore_index=True).sort_values(["patient_id", "slide_id"])
        df.to_csv(out_dir / "feature_manifest.csv", index=False)
        return out_dir / "feature_manifest.csv"
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--svs_dir", type=Path, default=DEFAULT_SVS_DIR)
    parser.add_argument("--uni_weights", type=Path, default=DEFAULT_UNI_WEIGHTS)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pattern", default="*.svs")
    parser.add_argument("--max_slides", type=int, default=None)
    parser.add_argument("--max_patches", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--target_mpp", type=float, default=0.5)
    parser.add_argument("--stride_factor", type=float, default=1.0)
    parser.add_argument("--tissue_threshold", type=float, default=0.35)
    parser.add_argument("--mask_max_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["pt", "h5"], default="pt")
    parser.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
    compression = None if args.compression == "none" else args.compression

    log0(rank, f"Found {len(slides)} slides. world_size={world}. out_dir={args.out_dir}")
    print(f"[rank {rank}] device={device} slides={len(my_slides)}", flush=True)

    uni = load_uni2_encoder(args.uni_weights, device)
    rows = []
    errors = []

    for slide_i, slide_path in enumerate(tqdm(my_slides, desc=f"rank{rank}")):
        out_path = args.out_dir / f"{slide_path.stem}.{args.format}"
        if not args.overwrite and is_complete_feature_file(out_path):
            if out_path.suffix.lower() in {".pt", ".pth"}:
                obj = torch.load(out_path, map_location="cpu")
                feat = obj["features"] if isinstance(obj, dict) else obj
                n_patches = int(feat.shape[0])
            else:
                with h5py.File(out_path, "r") as h:
                    n_patches = int(h["features"].shape[0])
            rows.append(
                {
                    "slide_id": slide_path.stem,
                    "patient_id": patient_id_from_slide(slide_path),
                    "svs_path": str(slide_path),
                    "feature_path": str(out_path),
                    "n_patches": n_patches,
                    "status": "skipped_existing",
                }
            )
            continue

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
            save_feature_file(
                out_path,
                slide_path=slide_path,
                features=features,
                coords=coords,
                read_size=read_size,
                native_mpp=native_mpp,
                patch_size=args.patch_size,
                target_mpp=args.target_mpp,
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
