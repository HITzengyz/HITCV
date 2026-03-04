import torch
from PIL import Image

import datasets.transforms as T


def _marker_xy(image):
    if torch.is_tensor(image):
        if image.dim() == 3:
            mask = image[0] > 0.5
        else:
            mask = image > 0.5
        ys, xs = torch.where(mask)
        assert ys.numel() == 1, "expected one marker pixel"
        return int(xs[0].item()), int(ys[0].item())
    arr = torch.from_numpy(__import__("numpy").array(image))
    if arr.dim() == 3:
        mask = arr[..., 0] > 0
    else:
        mask = arr > 0
    ys, xs = torch.where(mask)
    assert ys.numel() == 1, "expected one marker pixel"
    return int(xs[0].item()), int(ys[0].item())


def main():
    w, h = 8, 8
    marker = (2, 3)

    rgb = Image.new("RGB", (w, h), color=(0, 0, 0))
    rgb.putpixel(marker, (255, 255, 255))

    tir = Image.new("L", (w, h), color=0)
    tir.putpixel(marker, 255)

    radar_k = torch.zeros((4, h, w), dtype=torch.float32)
    radar_k[0, marker[1], marker[0]] = 1.0

    target = {
        "tir": tir,
        "tir_valid": torch.tensor(1.0),
        "radar_k": radar_k,
        "radar_valid": torch.tensor(1.0),
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "area": torch.zeros((0,), dtype=torch.float32),
        "iscrowd": torch.zeros((0,), dtype=torch.int64),
    }

    img, target = T.crop(rgb, target, (1, 1, 6, 6))
    img, target = T.hflip(img, target)
    img, target = T.pad(img, target, (1, 2))

    # Expected mapping:
    # crop (top=1,left=1): (2,3)->(1,2)
    # hflip width=6: x' = 6-1-1 = 4, y'=2
    # pad right/bottom: unchanged
    expected = (4, 2)

    rgb_xy = _marker_xy(img)
    tir_xy = _marker_xy(target["tir"])
    radar_xy = _marker_xy(target["radar_k"][0])

    assert rgb_xy == expected, f"RGB marker mismatch: {rgb_xy} vs {expected}"
    assert tir_xy == expected, f"TIR marker mismatch: {tir_xy} vs {expected}"
    assert radar_xy == expected, f"Radar marker mismatch: {radar_xy} vs {expected}"

    print("Deterministic RGB/TIR/Radar transform alignment OK.")


if __name__ == "__main__":
    main()
