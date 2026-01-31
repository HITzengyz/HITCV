import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from main import get_args_parser
from models import build_model
from util.misc import nested_tensor_from_tensor_list
import datasets.coco as coco


def _build_label_map(ann_path: Path):
    if not ann_path.exists():
        return {}
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    categories = data.get("categories", [])
    return {int(cat["id"]): str(cat.get("name", cat["id"])) for cat in categories}


def _color_for_label(label_id: int):
    # deterministic color from label id
    r = (label_id * 37) % 255
    g = (label_id * 67) % 255
    b = (label_id * 97) % 255
    return (r, g, b)


def _draw_boxes(image, boxes, scores, labels, label_map, score_thr):
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(boxes, scores, labels):
        if score.item() < score_thr:
            continue
        x0, y0, x1, y1 = box.tolist()
        label_id = int(label.item())
        color = _color_for_label(label_id)
        name = label_map.get(label_id, str(label_id))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0, y0), f"{name}:{score:.2f}", fill=color)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_name", default="00001.jpg")
    parser.add_argument("--data_root", default="/root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/data/waterscenes-coco")
    parser.add_argument("--ckpt", default="/root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/detr/test/output/checkpoint.pth")
    parser.add_argument("--score_thr", type=float, default=0.3)
    parser.add_argument("--out_dir", default="/root/autodl-tmp/DrivingSDK/model_examples/Deformable-DETR/detr/result")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_path = data_root / "instances_train.json"
    label_map = _build_label_map(ann_path)

    # model args
    base_parser = get_args_parser()
    margs = base_parser.parse_args([])
    margs.device = "npu"
    margs.coco_path = str(data_root)
    margs.masks = False
    margs.tir_mean = 0.5
    margs.tir_std = 0.5

    model, _, postprocessors = build_model(margs)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(margs.device)
    model.eval()

    rgb_path = data_root / "image" / args.img_name
    tir_path = data_root / "CAM_IR" / args.img_name

    rgb = Image.open(rgb_path).convert("RGB")
    tir = Image.open(tir_path).convert("L") if tir_path.exists() else None
    target = {
        "tir": tir,
        "tir_valid": torch.tensor(1.0 if tir is not None else 0.0)
    }

    transform = coco.make_coco_transforms("val", margs)
    img_tensor, _ = transform(rgb, target)
    samples = nested_tensor_from_tensor_list([img_tensor.to(margs.device)])

    with torch.no_grad():
        outputs = model(samples)
        target_sizes = torch.tensor([[rgb.height, rgb.width]], device=margs.device)
        results = postprocessors["bbox"](outputs, target_sizes)[0]

    boxes = results["boxes"].cpu()
    scores = results["scores"].cpu()
    labels = results["labels"].cpu()

    rgb_vis = _draw_boxes(rgb.copy(), boxes, scores, labels, label_map, args.score_thr)
    rgb_out = out_dir / f"{Path(args.img_name).stem}_rgb_pred.jpg"
    rgb_vis.save(rgb_out)

    # TIR visualization (grayscale -> RGB for drawing)
    if tir is not None:
        tir_vis = tir.convert("RGB")
        tir_vis = _draw_boxes(tir_vis, boxes, scores, labels, label_map, args.score_thr)
        tir_out = out_dir / f"{Path(args.img_name).stem}_tir_pred.jpg"
        tir_vis.save(tir_out)
        print("Saved:", rgb_out, tir_out)
    else:
        print("Saved:", rgb_out, "(no TIR found)")


if __name__ == "__main__":
    main()
