#!/usr/bin/env python3
"""Extract CONCH features using the locked UNI2-h patch coordinates.

This keeps patients, slides, and patch coordinates identical to the UNI2-h
main analysis; only the encoder changes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "results" / "locked_release_20260902" / "patient_manifest.csv"
DEFAULT_UNI_FEATURE_DIR = ROOT / "tcga_stad_uni2h" / "TCGA-STAD" / "features"
DEFAULT_SVS_DIR = ROOT / "external_downloads" / "tcga_stad" / "locked_246_svs"
DEFAULT_OUT = ROOT / "results" / "second_encoder_features_20x256" / "CONCH" / "TCGA-STAD" / "features"


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_info() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def find_svs(slide_id: str, svs_dirs: list[Path]) -> Path | None:
    names = [f"{slide_id}.svs", f"{slide_id}.tif", f"{slide_id}.tiff"]
    for d in svs_dirs:
        for name in names:
            p = d / name
            if p.exists() and p.stat().st_size > 0:
                return p
    for d in svs_dirs:
        hits = sorted(d.glob(f"{slide_id}.*"))
        hits = [p for p in hits if p.suffix.lower() in {".svs", ".tif", ".tiff"} and p.stat().st_size > 0]
        if hits:
            return hits[0]
    return None


def read_coords(slide_id: str, uni_feature_dir: Path, max_patches: int, seed: int) -> np.ndarray:
    h5_path = uni_feature_dir / f"{slide_id}.h5"
    with h5py.File(h5_path, "r") as h:
        if "coords_patching" in h:
            coords = h["coords_patching"][:]
        else:
            coords_ds = h["coords"]
            coords = coords_ds[0] if len(coords_ds.shape) == 3 else coords_ds[:]
    coords = np.asarray(coords, dtype=np.int64)
    if max_patches and len(coords) > max_patches:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), max_patches, replace=False)
        idx.sort()
        coords = coords[idx]
    return coords


def load_conch_model(args: argparse.Namespace, device: torch.device):
    token = args.hf_token or os.environ.get(args.hf_token_env, "")
    if args.backend in {"auto", "conch"}:
        try:
            from conch.open_clip_custom import create_model_from_pretrained

            source = args.checkpoint_path if args.checkpoint_path else args.hf_repo
            if source.startswith("hf_hub:"):
                model, preprocess = create_model_from_pretrained(
                    args.model_name,
                    source,
                    hf_auth_token=token or None,
                )
            else:
                model, preprocess = create_model_from_pretrained(
                    args.model_name,
                    checkpoint_path=source,
                )
            model.to(device).eval()

            def encode(batch: torch.Tensor) -> torch.Tensor:
                try:
                    return model.encode_image(batch, proj_contrast=False, normalize=False)
                except TypeError:
                    return model.encode_image(batch)

            return model, preprocess, encode, "conch.open_clip_custom"
        except Exception as exc:
            if args.backend == "conch":
                raise
            print(f"CONCH package backend unavailable, falling back to timm: {type(exc).__name__}: {exc}", flush=True)

    import timm
    from timm.data import create_transform, resolve_model_data_config

    model = timm.create_model(args.hf_repo, pretrained=True)
    model.to(device).eval()
    config = resolve_model_data_config(model)
    preprocess = create_transform(**config, is_training=False)

    def encode(batch: torch.Tensor) -> torch.Tensor:
        if hasattr(model, "encode_image"):
            try:
                return model.encode_image(batch, proj_contrast=False, normalize=False)
            except TypeError:
                return model.encode_image(batch)
        if hasattr(model, "forward_features"):
            out = model.forward_features(batch)
        else:
            out = model(batch)
        if isinstance(out, dict):
            for key in ["x_norm_clstoken", "pooled", "features", "last_hidden_state"]:
                if key in out:
                    out = out[key]
                    break
            if isinstance(out, dict):
                out = next(iter(out.values()))
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 3:
            out = out[:, 0]
        return out

    return model, preprocess, encode, "timm"


def is_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with h5py.File(path, "r") as h:
            feat = h["features"]
            return len(feat.shape) == 3 and feat.shape[0] == 1 and feat.shape[1] > 0 and feat.shape[2] > 0
    except Exception:
        return False


def save_h5(path: Path, features: np.ndarray, coords: np.ndarray, meta: dict[str, Any], compression: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(tmp, "w") as h:
        h.create_dataset("features", data=features[None, :, :].astype(np.float32), compression=compression)
        h.create_dataset("coords_patching", data=coords.astype(np.int64), compression=compression)
        h.create_dataset("coords", data=coords[None, :, :].astype(np.int64), compression=compression)
        for k, v in meta.items():
            h.attrs[k] = v
    tmp.replace(path)


@torch.inference_mode()
def extract_one(
    slide_id: str,
    patient_id: str,
    svs_path: Path,
    coords: np.ndarray,
    preprocess,
    encode,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    import openslide

    slide = openslide.OpenSlide(str(svs_path))
    feats = []
    read_size = int(round(args.patch_size_20x * args.read_mpp / args.target_mpp))
    read_size = max(1, read_size)
    resample = Image.Resampling.BICUBIC
    use_amp = device.type == "cuda"
    for start in range(0, len(coords), args.batch_size):
        batch_coords = coords[start : start + args.batch_size]
        imgs = []
        for x, y in batch_coords:
            img = slide.read_region((int(x), int(y)), 0, (read_size, read_size)).convert("RGB")
            if read_size != args.patch_size_20x:
                img = img.resize((args.patch_size_20x, args.patch_size_20x), resample)
            imgs.append(img)
        batch = torch.stack([preprocess(img) for img in imgs], dim=0).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            z = encode(batch)
        feats.append(z.detach().float().cpu().numpy())
    slide.close()
    arr = np.concatenate(feats, axis=0).astype(np.float32)
    meta = {
        "encoder": "CONCH",
        "slide_id": slide_id,
        "patient_id": patient_id,
        "svs_path": str(svs_path),
        "feature_dim": int(arr.shape[1]),
        "n_patches": int(arr.shape[0]),
        "coords_source": "UNI2-h coords_patching",
        "target_mpp": float(args.target_mpp),
        "read_mpp": float(args.read_mpp),
        "patch_size_20x": int(args.patch_size_20x),
        "model_name": args.model_name,
        "hf_repo": args.hf_repo,
        "checkpoint_path": args.checkpoint_path or "",
        "checkpoint_sha256": sha256(Path(args.checkpoint_path)) if args.checkpoint_path else None,
    }
    return arr, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--uni_feature_dir", type=Path, default=DEFAULT_UNI_FEATURE_DIR)
    parser.add_argument("--svs_dir", type=Path, default=DEFAULT_SVS_DIR)
    parser.add_argument("--extra_svs_dir", action="append", type=Path, default=[])
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--backend", choices=["auto", "conch", "timm"], default="auto")
    parser.add_argument("--model_name", default="conch_ViT-B-16")
    parser.add_argument("--hf_repo", default="hf_hub:MahmoodLab/CONCH")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--hf_token", default="")
    parser.add_argument("--hf_token_env", default="HF_TOKEN")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_patches", type=int, default=0, help="0 keeps all locked UNI2-h coordinates.")
    parser.add_argument("--patch_size_20x", type=int, default=256)
    parser.add_argument("--target_mpp", type=float, default=0.5)
    parser.add_argument("--read_mpp", type=float, default=0.5, help="Assume locked coords were generated for 20x-equivalent level-0 reads.")
    parser.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, local_rank, world = rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        raise RuntimeError("CUDA is required for CONCH extraction. Run inside a GPU-visible environment.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression
    model, preprocess, encode, backend = load_conch_model(args, device)
    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    shard = df.iloc[rank::world].copy()
    svs_dirs = [args.svs_dir] + args.extra_svs_dir
    rows = []
    for r in tqdm(list(shard.itertuples()), desc=f"rank{rank}/{world}"):
        slide_id = r.slide_id
        patient_id = r.patient_id
        out_path = args.out_dir / f"{slide_id}.h5"
        rec = {
            "patient_id": patient_id,
            "slide_id": slide_id,
            "feature_path": str(out_path),
            "status": "",
            "n_patches": "",
            "feature_dim": "",
            "svs_path": "",
            "error": "",
        }
        if is_complete(out_path) and not args.overwrite:
            with h5py.File(out_path, "r") as h:
                rec["status"] = "skipped_existing"
                rec["n_patches"] = int(h["features"].shape[1])
                rec["feature_dim"] = int(h["features"].shape[2])
                rec["svs_path"] = h.attrs.get("svs_path", "")
            rows.append(rec)
            continue
        svs_path = find_svs(slide_id, svs_dirs)
        if svs_path is None:
            rec["status"] = "missing_svs"
            rec["error"] = f"not found under {[str(p) for p in svs_dirs]}"
            rows.append(rec)
            continue
        try:
            coords = read_coords(slide_id, args.uni_feature_dir, args.max_patches, args.seed)
            features, meta = extract_one(slide_id, patient_id, svs_path, coords, preprocess, encode, device, args)
            meta["backend"] = backend
            save_h5(out_path, features, coords, meta, compression)
            rec["status"] = "extracted"
            rec["n_patches"] = int(features.shape[0])
            rec["feature_dim"] = int(features.shape[1])
            rec["svs_path"] = str(svs_path)
        except Exception as exc:
            rec["status"] = "failed"
            rec["svs_path"] = str(svs_path)
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(rec)
    rank_manifest = args.out_dir.parent / f"feature_manifest.rank{rank}.csv"
    pd.DataFrame(rows).to_csv(rank_manifest, index=False)
    if world == 1:
        pd.DataFrame(rows).sort_values(["patient_id", "slide_id"]).to_csv(args.out_dir.parent / "feature_manifest.csv", index=False)
    config = {
        "encoder": "CONCH",
        "backend": backend,
        "model_name": args.model_name,
        "hf_repo": args.hf_repo,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_sha256": sha256(Path(args.checkpoint_path)) if args.checkpoint_path else None,
        "license": "CC-BY-NC-ND-4.0",
        "source": "MahmoodLab/CONCH",
        "coords_source": str(args.uni_feature_dir),
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "batch_size": args.batch_size,
        "max_patches": args.max_patches,
        "patch_size_20x": args.patch_size_20x,
        "target_mpp": args.target_mpp,
        "read_mpp": args.read_mpp,
    }
    (args.out_dir.parent / f"extraction_config.rank{rank}.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"rank": rank, "world": world, "manifest": str(rank_manifest), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
