#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from PIL import Image


PROBE_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def try_decode_tir(path):
    try:
        with Image.open(path) as im:
            im.load()
            if im.mode in ("RGB", "RGBA", "CMYK"):
                return False, f"bad-mode:{im.mode}"
            if im.mode not in ("L", "P", "I;16", "I;16B", "I;16L", "F"):
                return False, f"unsupported-mode:{im.mode}"
        return True, ""
    except Exception as e:
        return False, repr(e)


def build_tir_index(tir_root):
    index = {}
    int_stems = {}
    files = [p for p in tir_root.iterdir() if p.is_file()]
    for p in files:
        ext = p.suffix.lower()
        if ext not in PROBE_EXTS:
            continue
        stem = p.stem
        prev = index.get(stem)
        if prev is None or PROBE_EXTS.index(ext) < PROBE_EXTS.index(prev.suffix.lower()):
            index[stem] = p
        if stem.isdigit():
            k = int(stem)
            int_stems.setdefault(k, []).append(stem)
    return index, int_stems, files


def resolve_tir_path(rgb_file_name, coco_root, tir_index, img_info):
    tir_file = img_info.get("tir_file")
    if tir_file:
        p = Path(tir_file)
        if p.is_file():
            return p
        p2 = coco_root / p
        if p2.is_file():
            return p2
    stem = Path(rgb_file_name).stem
    p = tir_index.get(stem)
    if p is not None and p.is_file():
        return p
    tir_root = coco_root / "WaterScenes_Fake_IR_Final"
    for ext in PROBE_EXTS:
        c = tir_root / f"{stem}{ext}"
        if c.is_file():
            return c
        cu = tir_root / f"{stem}{ext.upper()}"
        if cu.is_file():
            return cu
    return None


def zero_padding_hint(stem, int_stems):
    if not stem.isdigit():
        return None
    k = int(stem)
    cands = int_stems.get(k, [])
    if not cands:
        return None
    return ",".join(sorted(cands))


def main():
    parser = argparse.ArgumentParser("Check TIR coverage for WaterScenes")
    parser.add_argument("--coco_root", type=str, required=True)
    parser.add_argument("--ann", type=str, default="instances_train.json")
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    coco_root = Path(args.coco_root)
    ann_path = coco_root / args.ann
    tir_root = coco_root / "CAM_IR"
    if not ann_path.is_file():
        raise FileNotFoundError(f"missing annotation: {ann_path}")
    if not tir_root.is_dir():
        raise FileNotFoundError(f"missing CAM_IR dir: {tir_root}")

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    tir_index, int_stems, tir_files = build_tir_index(tir_root)

    total_rgb = len(images)
    total_tir = len(tir_files)
    overlap_count = 0
    missing_pairs = []
    unreadable = []
    mismatches = []

    for info in images:
        rgb_name = info.get("file_name", "")
        p = resolve_tir_path(rgb_name, coco_root, tir_index, info)
        if p is None:
            stem = Path(rgb_name).stem
            missing_pairs.append((rgb_name, str(tir_root / f"{stem}<ext>")))
            hint = zero_padding_hint(stem, int_stems)
            if hint is not None:
                mismatches.append((rgb_name, hint))
            continue
        overlap_count += 1
        ok, err = try_decode_tir(p)
        if not ok:
            unreadable.append((rgb_name, str(p), err))

    overlap_ratio = (overlap_count / float(total_rgb)) if total_rgb else 0.0

    print(f"total_rgb={total_rgb}")
    print(f"total_tir={total_tir}")
    print(f"overlap_count={overlap_count}")
    print(f"overlap_ratio={overlap_ratio:.6f}")
    print(f"resolvable_but_unreadable_count={len(unreadable)}")
    print("")
    print(f"top_missing_pairs (N={args.top_n}):")
    for rgb, expect in missing_pairs[: args.top_n]:
        print(f"  {rgb} -> {expect}")
    print("")
    print(f"top_resolvable_but_unreadable (N={args.top_n}):")
    for rgb, p, err in unreadable[: args.top_n]:
        print(f"  {rgb} -> {p} | err={err}")
    print("")
    print(f"top_zero_padding_mismatch_hints (N={args.top_n}):")
    for rgb, hint in mismatches[: args.top_n]:
        print(f"  {rgb} -> candidate_stems={hint}")


if __name__ == "__main__":
    main()
