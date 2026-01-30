import argparse

import torch
from PIL import Image

import datasets.transforms as T


def run_case(has_tir: bool):
    rgb = Image.new("RGB", (16, 16), color=(128, 64, 32))
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "area": torch.tensor([100.0]),
        "iscrowd": torch.tensor([0]),
        "size": torch.tensor([16, 16]),
        "orig_size": torch.tensor([16, 16]),
    }
    if has_tir:
        target["tir"] = Image.new("L", (16, 16), color=200)
        target["tir_valid"] = torch.tensor(1.0)
    else:
        target["tir"] = None
        target["tir_valid"] = torch.tensor(0.0)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406, 0.5, 0.0],
                    [0.229, 0.224, 0.225, 0.5, 1.0]),
    ])
    img_t, _ = transform(rgb, target)

    assert img_t.shape[0] == 5, f"expected 5 channels, got {img_t.shape[0]}"
    assert torch.isfinite(img_t).all(), "non-finite values in normalized input"
    assert img_t.abs().max().item() < 20.0, "unexpectedly large values after normalization"

    expected_valid = 1.0 if has_tir else 0.0
    tir_valid = img_t[4]
    assert torch.allclose(tir_valid, torch.full_like(tir_valid, expected_valid)), (
        f"tir_valid channel mismatch: expected {expected_valid}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_tir", action="store_true", help="test missing TIR path")
    args = parser.parse_args()

    run_case(has_tir=not args.no_tir)
    if not args.no_tir:
        run_case(has_tir=False)
    print("TIR pipeline checks passed.")


if __name__ == "__main__":
    main()
