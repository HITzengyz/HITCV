import argparse

import torch

from models import build_model
from util.misc import nested_tensor_from_tensor_list
from main import get_args_parser


def main():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fusion", type=str, default="sum")
    args = parser.parse_args()

    args.radar_enabled = True
    args.radar_fusion = args.fusion

    device = torch.device(args.device)
    model, _, _ = build_model(args)
    model.to(device)
    model.eval()

    rgb = torch.rand(3, args.height, args.width, device=device)
    tir = torch.zeros(1, args.height, args.width, device=device)
    tir_valid = torch.zeros(1, args.height, args.width, device=device)
    radar = torch.rand(4, args.height, args.width, device=device)
    radar_valid = torch.ones(1, args.height, args.width, device=device)
    sample = torch.cat([rgb, tir, tir_valid, radar, radar_valid], dim=0)

    samples = nested_tensor_from_tensor_list([sample])
    with torch.no_grad():
        outputs = model(samples)
    print("Radar inference OK. pred_logits:", outputs["pred_logits"].shape)


if __name__ == "__main__":
    main()
