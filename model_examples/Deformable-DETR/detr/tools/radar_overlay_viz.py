import argparse
import json
import logging
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import datasets.transforms as T  # noqa: E402


LOG = logging.getLogger("radar_overlay_viz")


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_first_existing(calib_dir: Path, names):
    for name in names:
        candidate = calib_dir / name
        if candidate.is_file():
            return candidate
    return None


def _load_matrix(path: Path):
    if path is None:
        return None
    if path.suffix.lower() == ".npy":
        return np.load(str(path))
    try:
        return np.loadtxt(str(path), delimiter=",")
    except ValueError:
        return np.loadtxt(str(path))


def _parse_kitti_like_calib(path: Path):
    if path is None or not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    data = {}
    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, values = line.split(":", 1)
        nums = [float(x) for x in values.strip().split()]
        data[key.strip()] = nums
    if not data:
        return None
    return data


def load_calibration(calib_dir: Path):
    intrinsics_candidates = [
        "K.npy",
        "K.txt",
        "intrinsics.npy",
        "intrinsics.txt",
        "camera_intrinsics.npy",
        "camera_intrinsics.txt",
    ]
    extrinsics_candidates = [
        "radar_to_camera.npy",
        "radar_to_camera.txt",
        "radar_to_cam.npy",
        "radar_to_cam.txt",
        "extrinsics.npy",
        "extrinsics.txt",
        "T_radar_to_cam.npy",
        "T_radar_to_cam.txt",
        "T_radar_to_camera.npy",
        "T_radar_to_camera.txt",
    ]
    r_candidates = ["R.npy", "R.txt", "radar_R.npy", "radar_R.txt"]
    t_candidates = ["t.npy", "t.txt", "radar_t.npy", "radar_t.txt"]
    single_txt = next(iter(calib_dir.glob("*.txt")), None)

    k_path = _find_first_existing(calib_dir, intrinsics_candidates)
    extr_path = _find_first_existing(calib_dir, extrinsics_candidates)
    r_path = _find_first_existing(calib_dir, r_candidates)
    t_path = _find_first_existing(calib_dir, t_candidates)

    K = _load_matrix(k_path)
    R = None
    t = None
    calib_meta = None

    if extr_path is not None:
        extr = _load_matrix(extr_path)
        if extr is not None:
            if extr.shape == (4, 4):
                R = extr[:3, :3]
                t = extr[:3, 3]
            elif extr.shape == (3, 4):
                R = extr[:3, :3]
                t = extr[:3, 3]
            elif extr.shape == (3, 3):
                R = extr
                if t_path is not None:
                    t = _load_matrix(t_path)
            else:
                LOG.warning("Unsupported extrinsics shape from %s: %s", extr_path, extr.shape)

    if R is None and r_path is not None:
        R = _load_matrix(r_path)
    if t is None and t_path is not None:
        t = _load_matrix(t_path)

    if (K is None or R is None or t is None) and single_txt is not None:
        calib_meta = _parse_kitti_like_calib(single_txt)
        if calib_meta is not None:
            if K is None and "t_camera_intrinsic" in calib_meta:
                k_vals = np.array(calib_meta["t_camera_intrinsic"], dtype=np.float32)
                if k_vals.size >= 9:
                    K = k_vals[:9].reshape(3, 3)
            if R is None and "t_camera_radar" in calib_meta:
                t_vals = np.array(calib_meta["t_camera_radar"], dtype=np.float32)
                if t_vals.size >= 16:
                    T = t_vals[:16].reshape(4, 4)
                    R = T[:3, :3]
                    t = T[:3, 3]

    calib_valid = True
    if K is None or K.shape != (3, 3):
        LOG.warning("Invalid intrinsics K (expected 3x3). Found: %s", None if K is None else K.shape)
        calib_valid = False
    if R is None or R.shape != (3, 3):
        LOG.warning("Invalid extrinsics R (expected 3x3). Found: %s", None if R is None else R.shape)
        calib_valid = False
    if t is None:
        LOG.warning("Missing extrinsics t (expected 3x1).")
        calib_valid = False
    else:
        t = np.asarray(t).reshape(-1)
        if t.shape[0] != 3:
            LOG.warning("Invalid extrinsics t (expected 3x1). Found: %s", t.shape)
            calib_valid = False
        t = t[:3]

    return {
        "K": K,
        "R": R,
        "t": t,
        "valid": calib_valid,
        "k_path": k_path,
        "extr_path": extr_path,
        "r_path": r_path,
        "t_path": t_path,
    }


def load_coco_images(ann_path: Path):
    if not ann_path.is_file():
        return {}, {}
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])
    by_id = {str(img.get("id")): img for img in images if "id" in img}
    by_name = {img.get("file_name"): img for img in images if "file_name" in img}
    return by_id, by_name


def extract_timestamp_from_name(name: str):
    if not name:
        return None
    matches = re.findall(r"(\d+\.\d+|\d+)", name)
    if not matches:
        return None
    return _parse_float(matches[-1])


def extract_radar_timestamp(path: Path):
    try:
        data = np.genfromtxt(str(path), delimiter=",", names=True, max_rows=1)
    except Exception:
        data = None
    if data is not None and data.dtype.names:
        names = {name.lower(): name for name in data.dtype.names}
        for key in ("timestamp", "time", "ts"):
            if key in names:
                val = data[names[key]]
                return _parse_float(val.item() if hasattr(val, "item") else val)
    return extract_timestamp_from_name(path.stem)


def list_radar_frames(radar_dir: Path):
    if not radar_dir.is_dir():
        return []
    frames = []
    for path in sorted(radar_dir.glob("*.csv")):
        frames.append({"path": path, "timestamp": extract_radar_timestamp(path)})
    return frames


def resolve_frame(frame_id: str, by_id, by_name, image_dir: Path):
    if frame_id in by_id:
        entry = by_id[frame_id]
        return entry.get("file_name"), entry
    if frame_id in by_name:
        entry = by_name[frame_id]
        return entry.get("file_name"), entry

    candidate = image_dir / frame_id
    if candidate.is_file():
        return candidate.name, {"file_name": candidate.name}

    stem = Path(frame_id).stem
    matches = list(image_dir.glob(f"{stem}.*"))
    if matches:
        return matches[0].name, {"file_name": matches[0].name}
    return None, None


def get_image_timestamp(entry):
    if entry is None:
        return None
    for key in ("timestamp", "time", "ts"):
        if key in entry:
            return _parse_float(entry.get(key))
    name = entry.get("file_name")
    return extract_timestamp_from_name(name)


def associate_radar_frames(rgb_ts, radar_frames, max_delta, accumulate_k):
    if rgb_ts is None or not radar_frames:
        return []
    candidates = []
    for frame in radar_frames:
        ts = frame.get("timestamp")
        if ts is None:
            continue
        delta = abs(ts - rgb_ts)
        if delta <= max_delta:
            candidates.append((delta, frame))
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0])
    return [frame for _, frame in candidates[:accumulate_k]]


def read_radar_points(path: Path):
    data = np.genfromtxt(str(path), delimiter=",", names=True)
    if data.size == 0:
        return None
    if data.ndim == 0:
        data = data.reshape(1)
    names = {name.lower(): name for name in data.dtype.names}
    if "range" not in names:
        LOG.warning("Radar CSV missing 'range' column: %s", path)
        return None
    if "azimuth" not in names:
        LOG.warning("Radar CSV missing 'azimuth' column: %s", path)
        return None

    ranges = data[names["range"]].astype(np.float32)
    azimuth = data[names["azimuth"]].astype(np.float32)
    elevation = data[names["elevation"]].astype(np.float32) if "elevation" in names else np.zeros_like(ranges)
    doppler = data[names["doppler"]].astype(np.float32) if "doppler" in names else None
    power = data[names["power"]].astype(np.float32) if "power" in names else None
    rcs = data[names["rcs"]].astype(np.float32) if "rcs" in names else None
    u = data[names["u"]].astype(np.float32) if "u" in names else None
    v = data[names["v"]].astype(np.float32) if "v" in names else None
    z = data[names["z"]].astype(np.float32) if "z" in names else None
    timestamp = None
    for key in ("timestamp", "time", "ts"):
        if key in names:
            timestamp = data[names[key]].astype(np.float64)
            break

    return {
        "range": ranges,
        "azimuth": azimuth,
        "elevation": elevation,
        "doppler": doppler,
        "power": power,
        "rcs": rcs,
        "u": u,
        "v": v,
        "z": z,
        "timestamp": timestamp,
    }


def convert_angles(values, unit):
    if values is None:
        return None
    if unit == "deg":
        return np.deg2rad(values)
    return values


def spherical_to_cartesian(rng, azimuth, elevation):
    x = rng * np.cos(elevation) * np.cos(azimuth)
    y = rng * np.cos(elevation) * np.sin(azimuth)
    z = rng * np.sin(elevation)
    return np.stack([x, y, z], axis=-1)


def transform_points(points, R, t):
    return points @ R.T + t[None, :]


def project_points(points, K):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    u = (K[0, 0] * x + K[0, 1] * y + K[0, 2] * z) / z
    v = (K[1, 0] * x + K[1, 1] * y + K[1, 2] * z) / z
    return np.stack([u, v, z], axis=-1)


def filter_points(uvz, image_size):
    w, h = image_size
    u = uvz[:, 0]
    v = uvz[:, 1]
    z = uvz[:, 2]
    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (z > 0)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    mask = valid & in_bounds
    return mask, valid


def validate_projection_ranges(uvz, image_size):
    w, h = image_size
    u = uvz[:, 0]
    v = uvz[:, 1]
    if u.size == 0 or v.size == 0:
        return
    if np.nanmax(u) > w * 5 or np.nanmin(u) < -w * 4:
        LOG.warning("Projection u range looks extreme: min=%s max=%s (w=%s)", np.nanmin(u), np.nanmax(u), w)
    if np.nanmax(v) > h * 5 or np.nanmin(v) < -h * 4:
        LOG.warning("Projection v range looks extreme: min=%s max=%s (h=%s)", np.nanmin(v), np.nanmax(v), h)


def select_color_values(radar, mode):
    if mode in ("doppler", "velocity"):
        if radar["doppler"] is None:
            return np.zeros_like(radar["range"]), "doppler"
        return radar["doppler"], "doppler"
    if mode == "power":
        if radar["power"] is None:
            return np.zeros_like(radar["range"]), "power"
        return radar["power"], "power"
    if mode == "rcs":
        if radar["rcs"] is None:
            return np.zeros_like(radar["range"]), "rcs"
        return radar["rcs"], "rcs"
    return radar["range"], "range"


def map_colors(values):
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    t = (values - vmin) / (vmax - vmin)
    t = np.clip(t, 0.0, 1.0)
    r = (255 * t).astype(np.uint8)
    g = (255 * (1.0 - np.abs(t - 0.5) * 2.0)).astype(np.uint8)
    b = (255 * (1.0 - t)).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def draw_points(image: Image.Image, uv, colors, radius):
    draw = ImageDraw.Draw(image)
    for (u, v), color in zip(uv, colors):
        x = int(round(u))
        y = int(round(v))
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=tuple(color))
    return image


def rasterize_revp(image_size, uv, radar, collision_score, score_channel):
    w, h = image_size
    revp = np.zeros((h, w, 4), dtype=np.float32)
    if uv.size == 0:
        return revp
    u = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, w - 1)
    v = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, h - 1)
    ranges = radar["range"]
    doppler = radar["doppler"] if radar["doppler"] is not None else np.zeros_like(ranges)
    power = radar["power"] if radar["power"] is not None else np.zeros_like(ranges)
    rcs = radar["rcs"] if radar["rcs"] is not None else np.zeros_like(ranges)

    for idx, (x, y) in enumerate(zip(u, v)):
        score = collision_score[idx]
        if score >= revp[y, x, score_channel]:
            revp[y, x, 0] = ranges[idx]
            revp[y, x, 1] = doppler[idx]
            revp[y, x, 2] = power[idx]
            revp[y, x, 3] = rcs[idx]
    return revp


def alpha_blend(base: Image.Image, overlay: Image.Image, alpha: float):
    return Image.blend(base, overlay, alpha)


def build_overlay_transforms(image_set: str):
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set == "train":
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
        ])
    if image_set == "val":
        return T.Compose([
            T.RandomResize([800], max_size=1333),
        ])
    raise ValueError(f"unknown transform set {image_set}")


def apply_transforms(image, points_uv, transform_set):
    if transform_set is None:
        return image, points_uv
    target = {"radar_points": torch.as_tensor(points_uv, dtype=torch.float32)}
    image_t, target_t = transform_set(image, target)
    pts = target_t.get("radar_points")
    if pts is None:
        pts = torch.zeros((0, 2), dtype=torch.float32)
    return image_t, pts.cpu().numpy()


def resolve_radar_files(frame_name, radar_dir: Path):
    if radar_dir is None:
        return None
    candidate = radar_dir / Path(frame_name).with_suffix(".csv")
    if candidate.is_file():
        return candidate
    return None


def process_frame(frame_id, args, by_id, by_name, radar_frames, transform_set):
    image_dir = args.data_root / "image"
    frame_name, entry = resolve_frame(frame_id, by_id, by_name, image_dir)
    if frame_name is None:
        raise FileNotFoundError(f"Frame not found: {frame_id}")

    rgb_path = image_dir / frame_name
    if not rgb_path.is_file():
        raise FileNotFoundError(f"RGB image missing: {rgb_path}")

    rgb_ts = get_image_timestamp(entry)
    radar_valid = 0
    calib_valid = args.calib["valid"]
    selected_frames = []
    radar_data = None
    delta = None

    if calib_valid and radar_frames:
        selected_frames = associate_radar_frames(rgb_ts, radar_frames, args.max_delta, args.accumulate_k)
        if selected_frames:
            delta = abs(selected_frames[0]["timestamp"] - rgb_ts) if rgb_ts is not None else None
            LOG.info("Selected radar frame: %s (delta=%.4f)", selected_frames[0]["path"].name, delta or -1.0)
        else:
            LOG.warning("No radar frame within max_delta=%.3f for %s", args.max_delta, frame_name)

    if calib_valid and not selected_frames:
        direct = resolve_radar_files(frame_name, args.radar_dir)
        if direct is not None:
            selected_frames = [{"path": direct, "timestamp": extract_radar_timestamp(direct)}]
            delta = None
            LOG.info("Using direct radar file match: %s", direct.name)

    if calib_valid and selected_frames:
        points = []
        attrs = {"doppler": [], "power": [], "rcs": [], "timestamp": []}
        for frame in selected_frames:
            radar = read_radar_points(frame["path"])
            if radar is None:
                continue
            points.append(radar)
        if points:
            radar_data = points
            radar_valid = 1

    image = Image.open(rgb_path).convert("RGB")

    if not calib_valid or radar_data is None:
        LOG.warning("Calibration invalid or radar missing for %s. Producing RGB-only overlay.", frame_name)
        return {
            "frame_name": frame_name,
            "image": image,
            "points_uv": np.zeros((0, 2), dtype=np.float32),
            "stats": {
                "total_points": 0,
                "in_fov": 0,
                "out_of_fov": 0,
                "dropped_ratio": 0.0,
                "radar_valid": 0,
                "calib_valid": int(calib_valid),
                "delta": delta,
            },
        }

    merged = radar_data[0]
    if len(radar_data) > 1:
        merged = {
            "range": np.concatenate([r["range"] for r in radar_data], axis=0),
            "azimuth": np.concatenate([r["azimuth"] for r in radar_data], axis=0),
            "elevation": np.concatenate([r["elevation"] for r in radar_data], axis=0),
            "doppler": None,
            "power": None,
            "rcs": None,
            "timestamp": None,
        }
        if any(r["doppler"] is not None for r in radar_data):
            merged["doppler"] = np.concatenate([r["doppler"] if r["doppler"] is not None else np.zeros_like(r["range"]) for r in radar_data], axis=0)
        if any(r["power"] is not None for r in radar_data):
            merged["power"] = np.concatenate([r["power"] if r["power"] is not None else np.zeros_like(r["range"]) for r in radar_data], axis=0)
        if any(r["rcs"] is not None for r in radar_data):
            merged["rcs"] = np.concatenate([r["rcs"] if r["rcs"] is not None else np.zeros_like(r["range"]) for r in radar_data], axis=0)

    if args.azimuth_unit:
        merged["azimuth"] = convert_angles(merged["azimuth"], args.azimuth_unit)
    if args.elevation_unit:
        merged["elevation"] = convert_angles(merged["elevation"], args.elevation_unit)

    use_uv = merged.get("u") is not None and merged.get("v") is not None
    if use_uv:
        LOG.info("Using pre-projected u/v from radar CSV.")
        z_vals = merged.get("z")
        if z_vals is None:
            z_vals = np.ones_like(merged["u"], dtype=np.float32)
        uvz = np.stack([merged["u"], merged["v"], z_vals], axis=-1)
    else:
        points_radar = spherical_to_cartesian(merged["range"], merged["azimuth"], merged["elevation"])
        points_cam = transform_points(points_radar, args.calib["R"], args.calib["t"])
        uvz = project_points(points_cam, args.calib["K"])
    validate_projection_ranges(uvz, image.size)

    mask, valid = filter_points(uvz, image.size)
    in_fov = int(mask.sum())
    total = int(len(uvz))
    out_of_fov = int((valid & ~mask).sum())
    dropped = int(total - in_fov)
    dropped_ratio = float(dropped / total) if total > 0 else 0.0

    uv = uvz[mask][:, :2]
    merged_filtered = {
        "range": merged["range"][mask],
        "azimuth": merged["azimuth"][mask],
        "elevation": merged["elevation"][mask],
        "doppler": merged["doppler"][mask] if merged["doppler"] is not None else None,
        "power": merged["power"][mask] if merged["power"] is not None else None,
        "rcs": merged["rcs"][mask] if merged["rcs"] is not None else None,
    }

    return {
        "frame_name": frame_name,
        "image": image,
        "points_uv": uv,
        "radar": merged_filtered,
        "stats": {
            "total_points": total,
            "in_fov": in_fov,
            "out_of_fov": out_of_fov,
            "dropped_ratio": dropped_ratio,
            "radar_valid": radar_valid,
            "calib_valid": int(calib_valid),
            "delta": delta,
        },
    }


def render_overlay(image, points_uv, radar, args):
    mode = args.render_mode
    values, used_mode = select_color_values(radar, mode)
    colors = map_colors(values) if values is not None else np.zeros((len(points_uv), 3), dtype=np.uint8)
    if args.order_by in ("power", "rcs"):
        if args.order_by == "power" and radar.get("power") is not None:
            order = np.argsort(radar["power"])
        elif args.order_by == "rcs" and radar.get("rcs") is not None:
            order = np.argsort(radar["rcs"])
        else:
            order = np.arange(len(points_uv))
        points_uv = points_uv[order]
        colors = colors[order]
        LOG.info("Point draw order: %s", args.order_by)
    return draw_points(image, points_uv, colors, args.point_radius), used_mode


def write_stats(stats_path: Path, stats, mode):
    payload = dict(stats)
    payload["render_mode"] = mode
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def render_revp_overlay(image, points_uv, radar, args):
    if points_uv.size == 0:
        return None
    if radar.get("power") is not None:
        collision_score = radar["power"]
        score_channel = 2
        LOG.info("REVP collision policy: max power")
    elif radar.get("rcs") is not None:
        collision_score = radar["rcs"]
        score_channel = 3
        LOG.info("REVP collision policy: max rcs")
    else:
        collision_score = radar["range"]
        score_channel = 0
        LOG.info("REVP collision policy: max range")

    revp = rasterize_revp(image.size, points_uv, radar, collision_score, score_channel)
    mode_values, _ = select_color_values(radar, args.render_mode)
    color_map = map_colors(mode_values)
    overlay = Image.new("RGB", image.size)
    if points_uv.size > 0:
        draw_points(overlay, points_uv, color_map, args.point_radius)
    return alpha_blend(image, overlay, args.revp_alpha)


def gather_frame_ids(args, by_id, by_name):
    if args.batch_list:
        with args.batch_list.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if args.batch_dir:
        return [p.name for p in sorted(args.batch_dir.glob("*.jpg"))]
    if args.frame_id:
        return [args.frame_id]
    if by_name:
        return list(by_name.keys())[:1]
    return []


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Project radar points onto RGB images for WaterScenes overlay debugging.\n"
            "Calibration file search order in --calib_dir:\n"
            "- Intrinsics: K.{npy,txt}, intrinsics.{npy,txt}, camera_intrinsics.{npy,txt}\n"
            "- Extrinsics: radar_to_camera.{npy,txt}, radar_to_cam.{npy,txt}, "
            "extrinsics.{npy,txt}, T_radar_to_cam.{npy,txt}, T_radar_to_camera.{npy,txt}\n"
            "- Optional: R.{npy,txt} and t.{npy,txt}"
        )
    )
    parser.add_argument("--frame_id", type=str, help="Image filename or COCO image id")
    parser.add_argument("--data_root", type=Path, default=Path("model_examples/Deformable-DETR/data/waterscenes-coco"))
    parser.add_argument("--out_dir", type=Path, default=Path("model_examples/Deformable-DETR/detr/result"))
    parser.add_argument("--render_mode", type=str, default="range", choices=("range", "doppler", "velocity", "power", "rcs"))
    parser.add_argument("--overlay_mode", type=str, default="raw", choices=("raw", "synced", "both"))
    parser.add_argument("--transform_set", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--calib_dir", type=Path, default=Path("model_examples/Deformable-DETR/data/waterscenes-coco/calib"))
    parser.add_argument("--radar_dir", type=Path, default=None)
    parser.add_argument("--max_delta", type=float, default=0.05)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--accumulate_k", type=int, default=1)
    parser.add_argument("--order_by", type=str, default="none", choices=("none", "power", "rcs"))
    parser.add_argument("--azimuth_unit", type=str, default="deg", choices=("deg", "rad"))
    parser.add_argument("--elevation_unit", type=str, default="deg", choices=("deg", "rad"))
    parser.add_argument("--revp", action="store_true", help="Also render REVP rasterization overlay")
    parser.add_argument("--revp_alpha", type=float, default=0.4)
    parser.add_argument("--stats_json", action="store_true", help="Write per-frame stats JSON")
    parser.add_argument("--batch_list", type=Path, default=None)
    parser.add_argument("--batch_dir", type=Path, default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--video_out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=100)
    return parser


def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_deterministic(args.seed)

    if args.accumulate_k > 1:
        LOG.warning("accumulate_k=%d without motion compensation may introduce ghosting.", args.accumulate_k)

    if args.radar_dir is None:
        args.radar_dir = args.data_root / "radar"
    if not args.radar_dir.exists():
        LOG.warning("Radar directory not found: %s", args.radar_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.calib = load_calibration(args.calib_dir)
    if not args.calib["valid"]:
        LOG.warning("Calibration invalid or missing. Overlay will be RGB-only.")

    ann_path = args.data_root / "instances_train.json"
    by_id, by_name = load_coco_images(ann_path)
    radar_frames = list_radar_frames(args.radar_dir)

    transform_set = None
    if args.overlay_mode in ("synced", "both"):
        transform_set = build_overlay_transforms(args.transform_set)
        LOG.info("Transform-synced overlay enabled (set=%s).", args.transform_set)
    LOG.info("Overlay mode: %s", args.overlay_mode)

    frame_ids = gather_frame_ids(args, by_id, by_name)
    if not frame_ids:
        LOG.error("No frames to process.")
        sys.exit(1)

    successes = 0
    failures = 0
    dropped_ratios = []
    overlay_paths = []

    for frame_id in frame_ids:
        try:
            result = process_frame(frame_id, args, by_id, by_name, radar_frames, transform_set)
            image = result["image"]
            points_uv = result["points_uv"]
            radar = result.get("radar")
            stats = result["stats"]

            outputs = []
            if args.overlay_mode in ("raw", "both"):
                if radar is None or points_uv.size == 0:
                    overlay = image.copy()
                    mode_used = "none"
                else:
                    overlay, mode_used = render_overlay(image.copy(), points_uv, radar, args)
                name = Path(result["frame_name"]).stem
                raw_path = args.out_dir / f"{name}_overlay.png"
                overlay.save(raw_path)
                outputs.append(("raw", raw_path, overlay.size, mode_used))
                overlay_paths.append(raw_path)
                LOG.info("Saved raw overlay: %s (mode=%s)", raw_path, mode_used)

            if args.overlay_mode in ("synced", "both"):
                synced_image, synced_points = apply_transforms(image.copy(), points_uv, transform_set)
                if radar is None or synced_points.size == 0:
                    overlay = synced_image.copy()
                    mode_used = "none"
                else:
                    overlay, mode_used = render_overlay(synced_image.copy(), synced_points, radar, args)
                name = Path(result["frame_name"]).stem
                synced_path = args.out_dir / f"{name}_overlay_synced.png"
                overlay.save(synced_path)
                outputs.append(("synced", synced_path, overlay.size, mode_used))
                overlay_paths.append(synced_path)
                LOG.info("Saved synced overlay: %s (mode=%s)", synced_path, mode_used)

            if len(outputs) == 2 and outputs[0][2] != outputs[1][2]:
                LOG.warning("Raw vs synced overlay sizes differ for %s: %s vs %s",
                            frame_id, outputs[0][2], outputs[1][2])

            if args.revp and radar is not None and points_uv.size > 0:
                revp_overlay = render_revp_overlay(image.copy(), points_uv, radar, args)
                if revp_overlay is not None:
                    name = Path(result["frame_name"]).stem
                    revp_path = args.out_dir / f"{name}_revp_overlay.png"
                    revp_overlay.save(revp_path)
                    overlay_paths.append(revp_path)

            if args.stats_json:
                name = Path(result["frame_name"]).stem
                stats_path = args.out_dir / f"{name}_stats.json"
                mode_label = outputs[0][3] if outputs else args.render_mode
                write_stats(stats_path, stats, mode_label)

            LOG.info("Frame %s: total=%d in_fov=%d out_of_fov=%d dropped_ratio=%.3f",
                     frame_id, stats["total_points"], stats["in_fov"], stats["out_of_fov"], stats["dropped_ratio"])
            successes += 1
            dropped_ratios.append(stats["dropped_ratio"])
        except Exception as exc:
            failures += 1
            LOG.exception("Failed to process frame %s: %s", frame_id, exc)
            if not args.continue_on_error:
                raise

    if dropped_ratios:
        mean_drop = float(np.mean(dropped_ratios))
    else:
        mean_drop = 0.0
    LOG.info("Batch summary: success=%d fail=%d mean_dropped_ratio=%.3f", successes, failures, mean_drop)

    if args.video_out and overlay_paths:
        try:
            import imageio  # noqa: WPS433
            frames = [np.array(Image.open(p)) for p in overlay_paths]
            imageio.mimsave(str(args.video_out), frames, fps=5)
            LOG.info("Saved video: %s", args.video_out)
        except Exception as exc:
            LOG.warning("Failed to write video (%s). Install imageio to enable.", exc)


if __name__ == "__main__":
    main()
