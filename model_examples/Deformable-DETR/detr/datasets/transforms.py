# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    if "tir" in target and target["tir"] is not None:
        target["tir"] = F.crop(target["tir"], *region)
    if "radar_k" in target and target["radar_k"] is not None:
        target["radar_k"] = F.crop(target["radar_k"], *region)

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    if "tir" in target and target["tir"] is not None:
        target["tir"] = F.hflip(target["tir"])
    if "radar_k" in target and target["radar_k"] is not None:
        target["radar_k"] = F.hflip(target["radar_k"])

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    if "tir" in target and target["tir"] is not None:
        target["tir"] = F.resize(target["tir"], size)
    if "radar_k" in target and target["radar_k"] is not None:
        target["radar_k"] = F.resize(target["radar_k"], size)

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    if "tir" in target and target["tir"] is not None:
        target["tir"] = F.pad(target["tir"], (0, 0, padding[0], padding[1]))
    if "radar_k" in target and target["radar_k"] is not None:
        target["radar_k"] = F.pad(target["radar_k"], (0, 0, padding[0], padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
        self.idx = 0
        self.length = len(self.sizes)

        seed = 100
        torch.manual_seed(seed)
        random.seed(seed)
        self.size = self.sizes[random.randint(0, self.length - 1)]

    def __call__(self, img, target=None):
        if self.idx % 8 == 0:
            self.size = self.sizes[random.randint(0, self.length - 1)]

        self.idx += 1
        return resize(img, target, self.size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            if self.transforms1 is None:
                return img, target
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        if target is None:
            return F.to_tensor(img), target
        target = target.copy()
        if "tir" in target and target["tir"] is not None and not torch.is_tensor(target["tir"]):
            target["tir"] = F.to_tensor(target["tir"])
        if "radar_k" in target and target["radar_k"] is not None and not torch.is_tensor(target["radar_k"]):
            target["radar_k"] = F.to_tensor(target["radar_k"])
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class ModalityDropout(object):
    def __init__(self, p=0.0):
        self.p = float(p)

    def __call__(self, image, target):
        if self.p <= 0:
            return image, target

        # 找到 valid
        tir_valid = None
        if "modalities" in target and "tir_valid" in target["modalities"]:
            tir_valid = target["modalities"]["tir_valid"]
        elif "tir_valid" in target:
            tir_valid = target["tir_valid"]

        # 如果本来就无效，直接返回
        if tir_valid is not None:
            tv = float(tir_valid.item()) if torch.is_tensor(tir_valid) else float(tir_valid)
            if tv <= 0:
                return image, target

        # 以概率 p 让 TIR 缺失
        if random.random() < self.p:
            # 置零 tir
            if "modalities" in target and "tir" in target["modalities"]:
                tir = target["modalities"]["tir"]
                if torch.is_tensor(tir):
                    target["modalities"]["tir"] = torch.zeros_like(tir)
                else:
                    target["modalities"]["tir"] = None
                # tir_valid -> 0
                tv = target["modalities"].get("tir_valid", None)
                target["modalities"]["tir_valid"] = torch.zeros_like(tv) if torch.is_tensor(tv) else torch.tensor(0.0)

            # 兼容 legacy keys
            if "tir" in target:
                tir = target["tir"]
                target["tir"] = torch.zeros_like(tir) if torch.is_tensor(tir) else None
            if "tir_valid" in target:
                tv = target["tir_valid"]
                target["tir_valid"] = torch.zeros_like(tv) if torch.is_tensor(tv) else torch.tensor(0.0)

        return image, target

#类似
'''class RadarModalityDropout(object):
    def __init__(self, p=0.0):
        self.p = float(p)

    def __call__(self, image, target):
        if self.p <= 0:
            return image, target

        radar_valid = None
        if "modalities" in target and "radar_valid" in target["modalities"]:
            radar_valid = target["modalities"]["radar_valid"]
        elif "radar_valid" in target:
            radar_valid = target["radar_valid"]

        if radar_valid is not None:
            rv = float(radar_valid.item()) if torch.is_tensor(radar_valid) else float(radar_valid)
            if rv <= 0:
                return image, target

        if random.random() < self.p:
            if "modalities" in target and "radar_k" in target["modalities"]:
                rk = target["modalities"]["radar_k"]
                target["modalities"]["radar_k"] = torch.zeros_like(rk)
                rv = target["modalities"].get("radar_valid", None)
                target["modalities"]["radar_valid"] = torch.zeros_like(rv) if torch.is_tensor(rv) else torch.tensor(0.0)

            if "radar_k" in target:
                rk = target["radar_k"]
                target["radar_k"] = torch.zeros_like(rk)
            if "radar_valid" in target:
                rv = target["radar_valid"]
                target["radar_valid"] = torch.zeros_like(rv) if torch.is_tensor(rv) else torch.tensor(0.0)

        return image, target'''


class Normalize(object):
    def __init__(self, mean, std, fusion_order=None):
        self.mean = mean
        self.std = std
        self.fusion_order = fusion_order or ["rgb", "tir", "tir_valid"]

    @staticmethod
    def _scalar_valid_to_map(valid_value, h, w, dtype, device):
        if valid_value is None:
            raise AssertionError("missing modality validity flag")
        if torch.is_tensor(valid_value):
            if valid_value.numel() == 1:
                value = float(valid_value.item())
            else:
                value = float(valid_value.float().mean().item())
        else:
            value = float(valid_value)
        value = 1.0 if value >= 0.5 else 0.0
        return torch.full((1, h, w), value, dtype=dtype, device=device)

    def __call__(self, image, target=None):
        if target is not None and any(k in self.fusion_order for k in ("tir", "tir_valid", "radar_k", "radar_valid")):
            target = target.copy()
            h, w = image.shape[-2:]
            parts = []
            modalities = {}
            for key in self.fusion_order:
                if key == "rgb":
                    parts.append(image)
                    modalities["rgb"] = image
                elif key == "tir":
                    tir = target.pop("tir", None)
                    if tir is None:
                        tir = torch.zeros((1, h, w), dtype=image.dtype, device=image.device)
                    parts.append(tir)
                    modalities["tir"] = tir
                elif key == "tir_valid":
                    tir_valid = target.pop("tir_valid", None)
                    tir_valid_map = self._scalar_valid_to_map(tir_valid, h, w, image.dtype, image.device)
                    parts.append(tir_valid_map)
                    modalities["tir_valid"] = tir_valid_map
                    target["tir_valid"] = tir_valid_map
                elif key == "radar_k":
                    radar_k = target.pop("radar_k", None)
                    if radar_k is None:
                        radar_k = torch.zeros((4, h, w), dtype=image.dtype, device=image.device)
                    parts.append(radar_k)
                    modalities["radar_k"] = radar_k
                elif key == "radar_valid":
                    radar_valid = target.pop("radar_valid", None)
                    radar_valid_map = self._scalar_valid_to_map(radar_valid, h, w, image.dtype, image.device)
                    parts.append(radar_valid_map)
                    modalities["radar_valid"] = radar_valid_map
                    target["radar_valid"] = radar_valid_map
                else:
                    raise AssertionError(f"unsupported fusion key: {key}")
            image = torch.cat(parts, dim=0)
            target["modalities"] = modalities

        if len(self.mean) != image.shape[0] or len(self.std) != image.shape[0]:
            raise AssertionError(
                f"normalize mean/std length mismatch: got mean={len(self.mean)} std={len(self.std)} "
                f"for channels={image.shape[0]}"
            )
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
