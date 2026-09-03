#!/usr/bin/env python3
"""Download/cache CONCH weights from Hugging Face without printing tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "external_models" / "CONCH" / "MahmoodLab_CONCH"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def file_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rows.append(
            {
                "path": str(p),
                "relative_path": str(p.relative_to(root)),
                "bytes": int(p.stat().st_size),
                "sha256": sha256(p),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="MahmoodLab/CONCH")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", ""))
    parser.add_argument("--token_env", default="HF_TOKEN")
    parser.add_argument("--env_file", type=Path, default=ROOT.parent / "appkey.env")
    parser.add_argument("--allow_pattern", action="append", default=None)
    parser.add_argument("--ignore_pattern", action="append", default=None)
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    from huggingface_hub import snapshot_download

    token = os.environ.get(args.token_env) or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token and not args.local_files_only:
        raise SystemExit(
            f"Missing Hugging Face token. Set {args.token_env}=... or put it in {args.env_file}. "
            "The token is never written to output files."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(args.out_dir),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=args.allow_pattern,
        ignore_patterns=args.ignore_pattern,
        force_download=args.force_download,
        local_files_only=args.local_files_only,
    )
    local_dir = Path(local_dir)
    rows = file_rows(local_dir)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "local_dir": str(local_dir),
        "n_files": len(rows),
        "total_bytes": int(sum(r["bytes"] for r in rows)),
        "files": rows,
        "token_recorded": False,
    }
    out_json = local_dir / "download_manifest.json"
    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["repo_id", "revision", "local_dir", "n_files", "total_bytes"]}, ensure_ascii=False))
    print(out_json)


if __name__ == "__main__":
    main()
