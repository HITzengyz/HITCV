import argparse

import torch

from main import get_args_parser
from models import build_model
from util.misc import nested_tensor_from_tensor_list


def _build_sample(case: str, h: int, w: int, device):
    rgb = torch.rand(3, h, w, device=device)
    tir = torch.rand(1, h, w, device=device)
    tir_valid = torch.ones(1, h, w, device=device)
    radar_k = torch.rand(4, h, w, device=device)
    radar_valid = torch.ones(1, h, w, device=device)

    if case == "no_tir":
        tir.zero_()
        tir_valid.zero_()
    elif case == "no_radar":
        radar_k.zero_()
        radar_valid.zero_()
    elif case == "both_missing":
        tir.zero_()
        tir_valid.zero_()
        radar_k.zero_()
        radar_valid.zero_()
    else:
        raise ValueError(f"unknown case: {case}")

    sample = torch.cat([rgb, tir, tir_valid, radar_k, radar_valid], dim=0)
    assert sample.shape[0] == 10, f"expected 10 channels, got {sample.shape[0]}"
    return sample


def _run_case(model, case: str, h: int, w: int, device):
    sample = _build_sample(case, h, w, device)
    tir_valid = sample[4]
    radar_valid = sample[9]
    if case in ("no_tir", "both_missing"):
        assert torch.all(tir_valid == 0), "tir_valid should be zeros"
    if case in ("no_radar", "both_missing"):
        assert torch.all(radar_valid == 0), "radar_valid should be zeros"

    samples = nested_tensor_from_tensor_list([sample])
    with torch.no_grad():
        outputs = model(samples)
    assert "pred_logits" in outputs and "pred_boxes" in outputs
    print(f"{case}: OK pred_logits={tuple(outputs['pred_logits'].shape)}")


def main():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    args.in_channels = 10
    args.use_waterscenes_modalities = True
    args.radar_channels = 4
    model, _, _ = build_model(args)
    model.to(device)
    model.eval()

    for case in ["no_tir", "no_radar", "both_missing"]:
        _run_case(model, case, args.height, args.width, device)

    print("Missing modality inference checks passed.")


if __name__ == "__main__":
    main()
