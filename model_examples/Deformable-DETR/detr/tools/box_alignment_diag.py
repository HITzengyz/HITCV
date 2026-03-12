import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.ops import box_iou

from main import get_args_parser
from models import build_model
from util.misc import nested_tensor_from_tensor_list
import datasets.coco as coco


def _color_for_label(label_id: int, bright=False):
    base = (label_id * 37) % 255
    r = (base + (80 if bright else 0)) % 255
    g = (base * 2 + (120 if bright else 0)) % 255
    b = (base * 3 + (160 if bright else 0)) % 255
    return (r, g, b)


def _draw_boxes(image, boxes, labels, color_override=None, width=2):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        x0, y0, x1, y1 = box.tolist()
        color = color_override or _color_for_label(int(label))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return image


def _denorm_rgb(img_tensor, mean, std):
    rgb = img_tensor[:3].clone()
    mean = torch.tensor(mean, dtype=rgb.dtype, device=rgb.device)[:, None, None]
    std = torch.tensor(std, dtype=rgb.dtype, device=rgb.device)[:, None, None]
    rgb = rgb * std + mean
    rgb = rgb.clamp(0.0, 1.0)
    rgb = (rgb * 255.0).byte().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(rgb)


def _cxcywh_to_xyxy_abs(boxes, size_hw):
    h, w = int(size_hw[0]), int(size_hw[1])
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    boxes = boxes * scale
    cx, cy, bw, bh = boxes.unbind(-1)
    x0 = cx - 0.5 * bw
    y0 = cy - 0.5 * bh
    x1 = cx + 0.5 * bw
    y1 = cy + 0.5 * bh
    return torch.stack([x0, y0, x1, y1], dim=-1)


def _max_iou_stats(gt_boxes, gt_labels, pred_boxes, pred_labels):
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return 0.0, 0.0
    iou = box_iou(gt_boxes, pred_boxes)
    max_any = float(iou.max().item())
    # same-class max
    max_same = 0.0
    for idx, label in enumerate(gt_labels):
        mask = pred_labels == label
        if mask.any():
            max_same = max(max_same, float(iou[idx, mask].max().item()))
    return max_any, max_same


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--score_thr", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--radar_enabled", action="store_true")
    parser.add_argument("--eval_use_size", action="store_true",
                        help="Scale predictions to target['size'] instead of orig_size")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_parser = get_args_parser()
    margs = base_parser.parse_args([])
    margs.device = args.device
    margs.coco_path = args.data_root
    margs.masks = False
    margs.num_classes = args.num_classes
    margs.radar_enabled = args.radar_enabled
    margs.debug_noobj = False
    margs.eval_use_size = args.eval_use_size

    dataset = coco.build("train", margs)
    transform = coco.make_coco_transforms("train", margs)

    model, _, postprocessors = build_model(margs)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
    model.to(margs.device)
    model.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    max_any_list = []
    max_same_list = []

    for idx in range(min(args.num_samples, len(dataset))):
        img, target = dataset[idx]
        rgb = _denorm_rgb(img, mean, std)

        size_hw = target["size"]
        orig_hw = target["orig_size"]
        h, w = img.shape[-2], img.shape[-1]
        print(f"[sizes] idx={idx} samples(H,W)=({h},{w}) "
              f"target[size]={tuple(size_hw.tolist())} "
              f"target[orig_size]={tuple(orig_hw.tolist())}")

        gt_boxes = _cxcywh_to_xyxy_abs(target["boxes"], size_hw)
        gt_labels = target["labels"].to(gt_boxes.device) if "labels" in target else torch.zeros((0,), dtype=torch.long)

        gt_img = _draw_boxes(rgb.copy(), gt_boxes.cpu(), gt_labels.cpu(), color_override=(0, 255, 0))
        scale_tag = "size" if args.eval_use_size else "orig"
        gt_img.save(out_dir / f"gt_{idx:02d}_scale-{scale_tag}.jpg")

        samples = nested_tensor_from_tensor_list([img.to(margs.device)])
        with torch.no_grad():
            outputs = model(samples)
            if args.eval_use_size:
                target_sizes = torch.tensor([size_hw.tolist()], device=margs.device)
            else:
                target_sizes = torch.tensor([orig_hw.tolist()], device=margs.device)
            results = postprocessors["bbox"](outputs, target_sizes)[0]

        scores = results["scores"].cpu()
        keep = scores >= args.score_thr
        pred_boxes = results["boxes"].cpu()[keep]
        pred_labels = results["labels"].cpu()[keep]

        pred_img = _draw_boxes(rgb.copy(), gt_boxes.cpu(), gt_labels.cpu(), color_override=(0, 255, 0))
        pred_img = _draw_boxes(pred_img, pred_boxes, pred_labels, color_override=(255, 0, 0))
        pred_img.save(out_dir / f"pred_{idx:02d}_scale-{scale_tag}.jpg")

        max_any, max_same = _max_iou_stats(gt_boxes.cpu(), gt_labels.cpu(), pred_boxes, pred_labels)
        max_any_list.append(max_any)
        max_same_list.append(max_same)

    if max_any_list:
        def _summary(vals, name):
            t = torch.tensor(vals)
            print(f"[iou] {name}: min={t.min():.3f}, med={t.median():.3f}, max={t.max():.3f}")
        _summary(max_any_list, "max_any")
        _summary(max_same_list, "max_same")


if __name__ == "__main__":
    main()
