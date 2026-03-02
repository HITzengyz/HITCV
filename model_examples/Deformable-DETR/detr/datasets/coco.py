# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import csv
import math
import os

import torch
import torch.utils.data
import torch.nn.functional as nnF
from pycocotools import mask as coco_mask
from PIL import Image
from torchvision.transforms import functional as TVF

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks,
        cache_mode=False,
        local_rank=0,
        local_size=1,
        tir_root=None,
        radar_root=None,
        calib_root=None,
        waterscenes_mode=False,
        radar_channels=4,
        tir_strict=True,
        tir_paired_only=False,
    ):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.tir_root = Path(tir_root) if tir_root is not None else None
        self.radar_root = Path(radar_root) if radar_root is not None else None
        self.calib_root = Path(calib_root) if calib_root is not None else None
        self.waterscenes_mode = bool(waterscenes_mode)
        self.radar_channels = int(radar_channels)
        self.tir_strict = bool(tir_strict)
        self.tir_paired_only = bool(tir_paired_only)
        self._tir_probe_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        self._tir_suffix_priority = {ext: i for i, ext in enumerate(self._tir_probe_exts)}
        self._tir_stem_index = {}
        if self.waterscenes_mode and self.tir_root is not None and self.tir_root.exists():
            for p in self.tir_root.iterdir():
                if not p.is_file():
                    continue
                ext_l = p.suffix.lower()
                if ext_l not in self._tir_suffix_priority:
                    continue
                stem = p.stem
                prev = self._tir_stem_index.get(stem)
                if prev is None:
                    self._tir_stem_index[stem] = p
                    continue
                if self._tir_suffix_priority[ext_l] < self._tir_suffix_priority[prev.suffix.lower()]:
                    self._tir_stem_index[stem] = p
        self._epoch = -1
        self._warned_once = set()
        self.tir_overlap_count = 0
        self.tir_overlap_ratio = 0.0
        self._filter_missing_images()
        self._compute_tir_overlap_stats()

    def _filter_missing_images(self):
        missing = 0
        dropped_unpaired = 0
        valid_ids = []
        for img_id in self.ids:
            info = self.coco.loadImgs(img_id)[0]
            file_name = info.get("file_name", "")
            img_path = file_name if os.path.isabs(file_name) else os.path.join(self.root, file_name)
            if not os.path.isfile(img_path):
                missing += 1
                continue
            if self.waterscenes_mode and self.tir_paired_only:
                tir_path = self.resolve_tir_path(file_name, img_info=info)
                if tir_path is None:
                    dropped_unpaired += 1
                    continue
            valid_ids.append(img_id)
        if missing:
            print(f"[WARN] Filtered {missing} missing images from dataset.")
        if dropped_unpaired:
            print(f"[WARN] Filtered {dropped_unpaired} unpaired RGB-only samples due to --tir_paired_only.")
        if missing or dropped_unpaired:
            self.ids = valid_ids

    def _compute_tir_overlap_stats(self):
        if not self.waterscenes_mode:
            return
        total = len(self.ids)
        if total <= 0:
            self.tir_overlap_count = 0
            self.tir_overlap_ratio = 0.0
            return
        overlap = 0
        for img_id in self.ids:
            info = self.coco.loadImgs(img_id)[0]
            if self.resolve_tir_path(info.get("file_name", ""), img_info=info) is not None:
                overlap += 1
        self.tir_overlap_count = overlap
        self.tir_overlap_ratio = float(overlap) / float(total)

    def set_epoch(self, epoch):
        epoch = int(epoch)
        if epoch != self._epoch:
            self._epoch = epoch
            self._warned_once.clear()

    def _warn_once(self, tag, message):
        key = (self._epoch, tag)
        if key in self._warned_once:
            return
        self._warned_once.add(key)
        print(f"[WARN] {message}")

    def _resolve_optional_path(self, value, fallback):
        if value:
            p = Path(value)
            if p.is_file():
                return str(p)
            p2 = Path(self.root) / p
            if p2.is_file():
                return str(p2)
            return None
        if fallback is not None and Path(fallback).is_file():
            return str(fallback)
        return None

    def resolve_tir_path(self, rgb_file_name, img_info=None):
        # Non-WaterScenes behavior remains unchanged.
        if not self.waterscenes_mode:
            tir_file = img_info.get("tir_file") if img_info is not None else None
            fallback = self.tir_root / rgb_file_name if self.tir_root is not None else None
            return self._resolve_optional_path(tir_file, fallback)

        tir_file = img_info.get("tir_file") if img_info is not None else None
        if tir_file:
            p = Path(tir_file)
            if p.is_file():
                return str(p)
            p2 = Path(self.root) / p
            if p2.is_file():
                return str(p2)
        if self.tir_root is None:
            return None
        stem = Path(rgb_file_name).stem
        indexed = self._tir_stem_index.get(stem)
        if indexed is not None and indexed.is_file():
            return str(indexed)
        # Extension probing order is case-insensitive and deterministic.
        for ext in self._tir_probe_exts:
            cand = self.tir_root / f"{stem}{ext}"
            if cand.is_file():
                return str(cand)
            cand_u = self.tir_root / f"{stem}{ext.upper()}"
            if cand_u.is_file():
                return str(cand_u)
        return None

    def _read_tir_tensor(self, tir_path):
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

    def _parse_calib_matrix(self, calib_path):
        extrinsic = None
        intrinsic = None
        with open(calib_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                name, values = line.split(":", 1)
                nums = [float(x) for x in values.strip().split()]
                if name.strip() == "t_camera_radar":
                    if len(nums) != 16:
                        raise ValueError(f"invalid t_camera_radar len={len(nums)}")
                    extrinsic = torch.tensor(nums, dtype=torch.float32).view(4, 4)
                elif name.strip() == "t_camera_intrinsic":
                    if len(nums) == 9:
                        intrinsic = torch.tensor(nums, dtype=torch.float32).view(3, 3)
                    elif len(nums) == 12:
                        k34 = torch.tensor(nums, dtype=torch.float32).view(3, 4)
                        if not torch.allclose(k34[:, 3], torch.zeros(3), atol=1e-6):
                            raise ValueError("t_camera_intrinsic 3x4 last column is non-zero")
                        intrinsic = k34[:, :3]
                    else:
                        raise ValueError(f"invalid t_camera_intrinsic len={len(nums)}")
        if extrinsic is None or intrinsic is None:
            raise ValueError("missing required calibration matrices")
        return extrinsic, intrinsic

    def _load_radar_points(self, radar_path):
        required = {"x", "y", "z", "doppler", "power"}
        points = []
        with open(radar_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("empty radar csv")
            cols = {c.strip() for c in reader.fieldnames}
            missing = sorted(required - cols)
            if missing:
                raise ValueError(f"missing radar columns: {missing}")
            for row in reader:
                try:
                    x = float(row["x"])
                    y = float(row["y"])
                    z = float(row["z"])
                    d = float(row["doppler"])
                    p = float(row["power"])
                except Exception:
                    continue
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(d) and math.isfinite(p)):
                    continue
                points.append((x, y, z, d, p))
        if not points:
            raise ValueError("no valid radar points")
        return torch.tensor(points, dtype=torch.float32)

    def _build_radar_map(self, radar_points, t_camera_radar, k, width, height):
        radar_xyz = radar_points[:, :3]
        doppler = radar_points[:, 3]
        power = radar_points[:, 4]

        n = radar_xyz.shape[0]
        ones = torch.ones((n, 1), dtype=torch.float32)
        xyz1 = torch.cat([radar_xyz, ones], dim=1)
        cam_xyz1 = torch.matmul(t_camera_radar, xyz1.t()).t()
        cam_xyz = cam_xyz1[:, :3]

        z = cam_xyz[:, 2]
        positive_depth = z > 1e-6
        if not torch.any(positive_depth):
            return torch.zeros((self.radar_channels, height, width), dtype=torch.float32), True

        cam_xyz = cam_xyz[positive_depth]
        doppler = doppler[positive_depth]
        power = power[positive_depth]

        uvw = torch.matmul(k, cam_xyz.t()).t()
        w = uvw[:, 2]
        valid_w = w.abs() > 1e-6
        if not torch.any(valid_w):
            return torch.zeros((self.radar_channels, height, width), dtype=torch.float32), True

        uvw = uvw[valid_w]
        cam_xyz = cam_xyz[valid_w]
        doppler = doppler[valid_w]
        power = power[valid_w]
        u = uvw[:, 0] / uvw[:, 2]
        v = uvw[:, 1] / uvw[:, 2]

        in_bounds = (
            torch.isfinite(u) & torch.isfinite(v) &
            (u >= 0) & (u < width) & (v >= 0) & (v < height)
        )
        if not torch.any(in_bounds):
            return torch.zeros((self.radar_channels, height, width), dtype=torch.float32), True

        u = u[in_bounds].long()
        v = v[in_bounds].long()
        cam_xyz = cam_xyz[in_bounds]
        doppler = doppler[in_bounds]
        power = power[in_bounds]
        cam_range = torch.linalg.norm(cam_xyz, dim=1)

        occ = torch.zeros((height, width), dtype=torch.float32)
        range_min = torch.full((height, width), float("inf"), dtype=torch.float32)
        doppler_sum = torch.zeros((height, width), dtype=torch.float32)
        doppler_count = torch.zeros((height, width), dtype=torch.float32)
        power_max = torch.zeros((height, width), dtype=torch.float32)

        for i in range(u.shape[0]):
            x = u[i].item()
            y = v[i].item()
            occ[y, x] = 1.0
            r = cam_range[i].item()
            if r < range_min[y, x]:
                range_min[y, x] = r
            doppler_sum[y, x] += doppler[i]
            doppler_count[y, x] += 1.0
            power_max[y, x] = max(power_max[y, x].item(), power[i].item())

        range_map = torch.where(torch.isfinite(range_min), range_min, torch.zeros_like(range_min))
        doppler_mean = torch.where(doppler_count > 0, doppler_sum / doppler_count, torch.zeros_like(doppler_sum))

        radar_channels = torch.stack([occ, range_map, doppler_mean, power_max], dim=0)
        return radar_channels, True

    def _load_tir_with_valid(self, img_info, image_id, img):
        tir = None
        tir_valid = 0.0
        tir_file = self.resolve_tir_path(img_info.get("file_name", ""), img_info=img_info)
        if tir_file:
            try:
                if self.waterscenes_mode:
                    tir = self._read_tir_tensor(tir_file)
                    if (tir.shape[-2], tir.shape[-1]) != (img.height, img.width):
                        tir = nnF.interpolate(
                            tir.unsqueeze(0),
                            size=(img.height, img.width),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                else:
                    tir = self.get_tir(tir_file)
                    if tir is not None and tir.size != img.size:
                        tir = tir.resize(img.size, resample=Image.BILINEAR)
                if tir is not None:
                    tir_valid = 1.0
            except Exception as exc:
                if self.waterscenes_mode and self.tir_strict:
                    raise RuntimeError(
                        f"TIR strict mode failure: image_id={image_id}, file={tir_file}, err={repr(exc)}"
                    ) from exc
                self._warn_once(
                    "tir-read-failed",
                    f"TIR read failed for image_id={image_id}, file={tir_file}; fallback to zeros (tir_valid=0). err={repr(exc)}",
                )
                tir = None
                tir_valid = 0.0
        return tir, torch.tensor(tir_valid, dtype=torch.float32)

    def _load_radar_with_valid(self, img_info, image_id, width, height):
        radar = torch.zeros((self.radar_channels, height, width), dtype=torch.float32)
        radar_valid = torch.tensor(0.0, dtype=torch.float32)
        if not self.waterscenes_mode:
            return radar, radar_valid

        stem = Path(img_info.get("file_name", "")).stem
        radar_file = self._resolve_optional_path(
            img_info.get("radar_file"),
            self.radar_root / f"{stem}.csv" if self.radar_root is not None else None,
        )
        calib_file = self._resolve_optional_path(
            img_info.get("calib_file"),
            self.calib_root / f"{stem}.txt" if self.calib_root is not None else None,
        )
        if radar_file is None or calib_file is None:
            if radar_file is None:
                self._warn_once(
                    "radar-path-missing",
                    f"Radar file missing for image_id={image_id}; fallback to zeros (radar_valid=0).",
                )
            if calib_file is None:
                self._warn_once(
                    "calib-path-missing",
                    f"Calibration file missing for image_id={image_id}; fallback to zeros (radar_valid=0).",
                )
            return radar, radar_valid

        try:
            points = self._load_radar_points(radar_file)
            t_camera_radar, intrinsic = self._parse_calib_matrix(calib_file)
            radar, ok = self._build_radar_map(points, t_camera_radar, intrinsic, width, height)
            radar_valid = torch.tensor(1.0 if ok else 0.0, dtype=torch.float32)
        except Exception as exc:
            self._warn_once(
                "radar-read-failed",
                f"Radar/calib parse failed for image_id={image_id}, radar={radar_file}, calib={calib_file}; fallback to zeros (radar_valid=0). err={exc}",
            )
            radar = torch.zeros((self.radar_channels, height, width), dtype=torch.float32)
            radar_valid = torch.tensor(0.0, dtype=torch.float32)
        return radar, radar_valid

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        img_info = self.coco.loadImgs(image_id)[0]
        tir, tir_valid = self._load_tir_with_valid(img_info, image_id, img)
        radar_k, radar_valid = self._load_radar_with_valid(img_info, image_id, img.width, img.height)
        target["tir"] = tir
        target["tir_valid"] = tir_valid
        if self.waterscenes_mode:
            target["radar_k"] = radar_k
            target["radar_valid"] = radar_valid
            # Preserve a WaterScenes-local modality contract while reusing legacy keys in transforms.
            target["modalities"] = {
                "rgb": img,
                "tir": tir,
                "tir_valid": tir_valid,
                "radar_k": radar_k,
                "radar_valid": radar_valid,
            }
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args=None):

    if args is None:
        tir_mean = 0.5
        tir_std = 0.5
        tir_dropout = 0.0
        radar_dropout = 0.3
        radar_mean = [0.0, 0.0, 0.0, 0.0]
        radar_std = [1.0, 1.0, 1.0, 1.0]
        use_waterscenes_modalities = False
    else:
        tir_mean = float(getattr(args, "tir_mean", 0.5))
        tir_std = float(getattr(args, "tir_std", 0.5))
        tir_dropout = float(getattr(args, "tir_dropout", 0.0))
        radar_dropout = float(getattr(args, "radar_dropout", 0.3))
        radar_mean = list(getattr(args, "radar_mean", [0.0, 0.0, 0.0, 0.0]))
        radar_std = list(getattr(args, "radar_std", [1.0, 1.0, 1.0, 1.0]))
        use_waterscenes_modalities = bool(getattr(args, "use_waterscenes_modalities", False))

    if use_waterscenes_modalities:
        mean = [0.485, 0.456, 0.406, tir_mean, 0.0] + radar_mean + [0.0]
        std = [0.229, 0.224, 0.225, tir_std, 1.0] + radar_std + [1.0]
        fusion_order = ["rgb", "tir", "tir_valid", "radar_k", "radar_valid"]
    else:
        mean = [0.485, 0.456, 0.406, tir_mean, 0.0]
        std = [0.229, 0.224, 0.225, tir_std, 1.0]
        fusion_order = ["rgb", "tir", "tir_valid"]

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std, fusion_order=fusion_order)
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                None,
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                ])
            ),
            T.RandomResize(scales, max_size=1333),
            T.ModalityDropout(p=tir_dropout),
            T.RadarModalityDropout(p=radar_dropout),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ModalityDropout(p=tir_dropout),
            T.RadarModalityDropout(p=radar_dropout),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    coco_ann_dir = root / "annotations"
    coco_train = coco_ann_dir / f'{mode}_train2017.json'
    coco_val = coco_ann_dir / f'{mode}_val2017.json'
    waterscenes_train = root / "instances_train.json"
    waterscenes_val = root / "instances_val.json"
    tir_root = root / "WaterScenes_Fake_IR_Final"
    radar_root = root / "radar"
    calib_root = root / "calib"
    waterscenes_mode = False
    if coco_train.exists():
        PATHS = {
            "train": (root / "train2017", coco_train),
            "val": (root / "val2017", coco_val),
        }
        if not tir_root.exists():
            tir_root = None
        radar_root = None
        calib_root = None
        setattr(args, "use_waterscenes_modalities", False)
        setattr(args, "modality_order", ["rgb", "tir", "tir_valid"])
        setattr(args, "tir_valid_channel_idx", 4)
        setattr(args, "radar_valid_channel_idx", -1)
        setattr(args, "in_channels", 5)
    elif waterscenes_train.exists() and (root / "image").exists():
        PATHS = {
            "train": (root / "image", waterscenes_train),
            "val": (root / "image", waterscenes_val if waterscenes_val.exists() else waterscenes_train),
        }
        waterscenes_mode = True
        if not tir_root.exists():
            tir_root = None
        if not radar_root.exists():
            radar_root = None
        if not calib_root.exists():
            calib_root = None
        radar_channels = int(getattr(args, "radar_channels", 4))
        setattr(args, "use_waterscenes_modalities", True)
        setattr(args, "radar_channels", radar_channels)
        setattr(args, "modality_order", ["rgb", "tir", "tir_valid", "radar_k", "radar_valid"])
        setattr(args, "tir_valid_channel_idx", 4)
        setattr(args, "radar_valid_channel_idx", 5 + radar_channels)
        setattr(args, "in_channels", 5 + radar_channels + 1)
    else:
        raise FileNotFoundError(
            "Unsupported COCO layout. Expected either COCO-2017 layout "
            "or waterscenes-coco layout with image/ and instances_train.json."
        )

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set, args), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(),
                            tir_root=tir_root, radar_root=radar_root, calib_root=calib_root,
                            waterscenes_mode=waterscenes_mode, radar_channels=int(getattr(args, "radar_channels", 4)),
                            tir_strict=bool(int(getattr(args, "tir_strict", 1))),
                            tir_paired_only=bool(int(getattr(args, "tir_paired_only", 0))))
    return dataset
