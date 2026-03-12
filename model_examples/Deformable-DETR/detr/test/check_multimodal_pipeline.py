import torch
from PIL import Image

import datasets.transforms as T


def _run_case(has_tir: bool, has_radar: bool):
    rgb = Image.new("RGB", (16, 16), color=(120, 80, 40))
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "area": torch.tensor([100.0]),
        "iscrowd": torch.tensor([0]),
        "size": torch.tensor([16, 16]),
        "orig_size": torch.tensor([16, 16]),
    }
    target["tir"] = Image.new("L", (16, 16), color=200) if has_tir else None
    target["tir_valid"] = torch.tensor(1.0 if has_tir else 0.0)
    if has_radar:
        radar = torch.zeros((4, 16, 16), dtype=torch.float32)
        radar[0, 3, 4] = 1.0
        radar[1, 3, 4] = 5.0
        radar[2, 3, 4] = -0.8
        radar[3, 3, 4] = 10.0
        target["radar_k"] = radar
        target["radar_valid"] = torch.tensor(1.0)
    else:
        target["radar_k"] = torch.zeros((4, 16, 16), dtype=torch.float32)
        target["radar_valid"] = torch.tensor(0.0)

    fusion_order = ["rgb", "tir", "tir_valid", "radar_k", "radar_valid"]
    mean = [0.485, 0.456, 0.406, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    std = [0.229, 0.224, 0.225, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std, fusion_order=fusion_order),
    ])
    img_t, out_target = transform(rgb, target)

    assert img_t.shape[0] == 10, f"expected 10 channels, got {img_t.shape[0]}"
    assert torch.isfinite(img_t).all(), "non-finite values in fused input"
    assert len(mean) == img_t.shape[0] and len(std) == img_t.shape[0], "mean/std length mismatch"

    for idx in [4, 9]:
        uniq = torch.unique(img_t[idx].to(torch.float32))
        assert torch.all((uniq == 0.0) | (uniq == 1.0)), f"valid channel {idx} not binary: {uniq}"
        uniq_fp16 = torch.unique(img_t[idx].to(torch.float16).to(torch.float32))
        assert torch.all((uniq_fp16 == 0.0) | (uniq_fp16 == 1.0)), f"valid channel {idx} fp16 drift: {uniq_fp16}"
        uniq_bf16 = torch.unique(img_t[idx].to(torch.bfloat16).to(torch.float32))
        assert torch.all((uniq_bf16 == 0.0) | (uniq_bf16 == 1.0)), f"valid channel {idx} bf16 drift: {uniq_bf16}"

    assert "tir_valid" in out_target and "radar_valid" in out_target
    assert out_target["tir_valid"].shape[0] == 1 and out_target["radar_valid"].shape[0] == 1


def _expect_length_assert():
    rgb = Image.new("RGB", (8, 8), color=(0, 0, 0))
    target = {"tir": None, "tir_valid": torch.tensor(0.0)}
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456], [0.229, 0.224], fusion_order=["rgb", "tir", "tir_valid"]),
    ])
    try:
        transform(rgb, target)
    except AssertionError:
        return
    raise AssertionError("expected Normalize length mismatch assertion")


def main():
    _run_case(has_tir=True, has_radar=True)
    _run_case(has_tir=False, has_radar=True)
    _run_case(has_tir=True, has_radar=False)
    _run_case(has_tir=False, has_radar=False)
    _expect_length_assert()
    print("Multimodal normalization and validity sanity checks passed.")


if __name__ == "__main__":
    main()
