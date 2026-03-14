#!/usr/bin/env python3
"""Compute global mean/std for single-channel TIR images.

The TIR decoding and normalization strictly follow datasets/coco.py::_read_tir_tensor:
- L/P     -> float32 / 255
- I;16*   -> float32 / 65535
- F       -> float32 and clamp to [0, 1]

This script computes pixel-level global mean/std over all valid files, not per-image averaging.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torchvision.transforms import functional as TVF

_VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _collect_tir_files(tir_root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(tir_root):
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            if p.suffix.lower() in _VALID_EXTS:
                files.append(p)
    files.sort()
    return files


def _read_tir_tensor(tir_path: Path) -> torch.Tensor:
    with Image.open(tir_path) as im:
        im.load()
        if im.mode in ("RGB", "RGBA", "CMYK"):
            raise ValueError(f"expected single-channel TIR image, got mode={im.mode}")
        tir = TVF.pil_to_tensor(im)
        if tir.ndim != 3 or tir.shape[0] != 1:
            raise ValueError(f"expected 1xHxW tensor, got shape={tuple(tir.shape)}")
        if im.mode in ("L", "P"):
            tir = tir.to(dtype=torch.float32) / 255.0
        elif im.mode in ("I;16", "I;16B", "I;16L"):
            tir = tir.to(dtype=torch.float32) / 65535.0
        elif im.mode == "F":
            tir = tir.to(dtype=torch.float32).clamp_(0.0, 1.0)
        else:
            raise ValueError(f"unsupported TIR mode for normalization: {im.mode}")
    return tir


def _compute_stats(paths: List[Path]) -> Dict[str, float]:
    sum_v = 0.0
    sum_sq = 0.0
    num_pixels = 0
    num_images = 0

    for idx, path in enumerate(paths, start=1):
        try:
            tir = _read_tir_tensor(path)
        except Exception as exc:
            _warn(f"skip unreadable/invalid file: {path} ({repr(exc)})")
            continue

        tir64 = tir.to(dtype=torch.float64)
        sum_v += float(tir64.sum().item())
        sum_sq += float((tir64 * tir64).sum().item())
        num_pixels += int(tir64.numel())
        num_images += 1

        if idx % 1000 == 0:
            print(f"[INFO] scanned={idx}, valid_images={num_images}")

    if num_pixels == 0:
        raise RuntimeError("no valid TIR pixels found; cannot compute statistics")

    mean = sum_v / float(num_pixels)
    var = max(sum_sq / float(num_pixels) - mean * mean, 0.0)
    std = math.sqrt(var)

    return {
        "mean": mean,
        "std": std,
        "num_images": num_images,
        "num_pixels": num_pixels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute global TIR mean/std from a directory tree")
    parser.add_argument("--tir_root", required=True, type=str, help="Root directory of TIR images")
    parser.add_argument(
        "--save_path",
        default="",
        type=str,
        help="Optional path to save result json, e.g. ./tir_stats.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tir_root = Path(args.tir_root).expanduser().resolve()
    if not tir_root.exists() or not tir_root.is_dir():
        raise FileNotFoundError(f"tir_root does not exist or is not a directory: {tir_root}")

    paths = _collect_tir_files(tir_root)
    if not paths:
        raise RuntimeError(f"no candidate files found under {tir_root}; valid exts={sorted(_VALID_EXTS)}")

    print(f"[INFO] tir_root={tir_root}")
    print(f"[INFO] candidate_files={len(paths)}")
    result = _compute_stats(paths)

    output = {
        "tir_root": str(tir_root),
        "extensions": sorted(_VALID_EXTS),
        **result,
    }

    print("[RESULT]")
    print(json.dumps(output, indent=2, ensure_ascii=False))

    if args.save_path:
        save_path = Path(args.save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[INFO] saved: {save_path}")


if __name__ == "__main__":
    main()
