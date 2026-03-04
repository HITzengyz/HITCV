import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


def _load_viz_module():
    import importlib.util
    module_path = Path(__file__).resolve().parents[1] / "tools" / "radar_overlay_viz.py"
    spec = importlib.util.spec_from_file_location("radar_overlay_viz", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_coco_ann(path: Path, image_name: str):
    payload = {
        "images": [
            {"id": 1, "file_name": image_name, "timestamp": 1.0},
        ],
        "annotations": [],
        "categories": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_calib(calib_dir: Path):
    k = np.eye(3, dtype=np.float32)
    r = np.eye(3, dtype=np.float32)
    t = np.zeros((3,), dtype=np.float32)
    np.savetxt(str(calib_dir / "K.txt"), k)
    np.savetxt(str(calib_dir / "R.txt"), r)
    np.savetxt(str(calib_dir / "t.txt"), t)


class RadarOverlayVizTests(unittest.TestCase):
    def setUp(self):
        self.viz = _load_viz_module()

    def _build_args(self, data_root: Path, calib, radar_dir: Path):
        return SimpleNamespace(
            data_root=data_root,
            out_dir=data_root / "out",
            calib=calib,
            radar_dir=radar_dir,
            max_delta=0.05,
            accumulate_k=1,
            azimuth_unit="deg",
            elevation_unit="deg",
            render_mode="range",
            point_radius=2,
            order_by="none",
        )

    def test_missing_radar_file_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "image"
            image_dir.mkdir(parents=True)
            ann_path = root / "instances_train.json"
            image_name = "0001.jpg"
            _write_coco_ann(ann_path, image_name)

            img = Image.new("RGB", (10, 10), (0, 0, 0))
            img.save(image_dir / image_name)

            calib_dir = root / "calib"
            calib_dir.mkdir()
            _write_calib(calib_dir)
            calib = self.viz.load_calibration(calib_dir)

            radar_dir = root / "radar"
            radar_dir.mkdir()
            by_id, by_name = self.viz.load_coco_images(ann_path)
            radar_frames = []

            args = self._build_args(root, calib, radar_dir)
            result = self.viz.process_frame("1", args, by_id, by_name, radar_frames, None)

            self.assertEqual(result["stats"]["radar_valid"], 0)
            self.assertEqual(result["stats"]["total_points"], 0)
            out_path = args.out_dir / "0001_overlay.png"
            args.out_dir.mkdir(parents=True, exist_ok=True)
            result["image"].save(out_path)
            self.assertTrue(out_path.exists())

    def test_empty_radar_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "image"
            image_dir.mkdir(parents=True)
            ann_path = root / "instances_train.json"
            image_name = "0002.jpg"
            _write_coco_ann(ann_path, image_name)

            img = Image.new("RGB", (12, 8), (10, 10, 10))
            img.save(image_dir / image_name)

            calib_dir = root / "calib"
            calib_dir.mkdir()
            _write_calib(calib_dir)
            calib = self.viz.load_calibration(calib_dir)

            radar_dir = root / "radar"
            radar_dir.mkdir()
            radar_path = radar_dir / "0002.csv"
            radar_path.write_text("range,azimuth,elevation,doppler,power,rcs\n", encoding="utf-8")

            by_id, by_name = self.viz.load_coco_images(ann_path)
            radar_frames = self.viz.list_radar_frames(radar_dir)
            args = self._build_args(root, calib, radar_dir)

            result = self.viz.process_frame("0002.jpg", args, by_id, by_name, radar_frames, None)
            self.assertEqual(result["stats"]["total_points"], 0)
            self.assertEqual(result["stats"]["in_fov"], 0)
            self.assertEqual(result["stats"]["radar_valid"], 0)

    def test_missing_calibration_no_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "image"
            image_dir.mkdir(parents=True)
            ann_path = root / "instances_train.json"
            image_name = "0003.jpg"
            _write_coco_ann(ann_path, image_name)

            img = Image.new("RGB", (10, 6), (0, 0, 0))
            img.save(image_dir / image_name)

            calib = {"K": None, "R": None, "t": None, "valid": False}
            radar_dir = root / "radar"
            radar_dir.mkdir()
            by_id, by_name = self.viz.load_coco_images(ann_path)
            args = self._build_args(root, calib, radar_dir)

            result = self.viz.process_frame("0003.jpg", args, by_id, by_name, [], None)
            self.assertEqual(result["stats"]["calib_valid"], 0)
            self.assertEqual(result["stats"]["radar_valid"], 0)
            self.assertEqual(result["stats"]["total_points"], 0)

    def test_horizontal_flip_points(self):
        from datasets import transforms as T

        img = Image.new("RGB", (10, 8), (0, 0, 0))
        points = np.array([[2.0, 3.0], [9.5, 0.0]], dtype=np.float32)
        _, target = T.hflip(img, {"radar_points": points})
        flipped = target["radar_points"].numpy()
        expected = np.array([[10.0 - 2.0, 3.0], [10.0 - 9.5, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(flipped, expected, rtol=1e-5, atol=1e-5)

    def test_projection_filters_invalid(self):
        uvz = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, -1.0],
            [np.nan, 1.0, 2.0],
            [9.0, 9.0, 0.1],
            [11.0, 1.0, 1.0],
        ], dtype=np.float32)
        mask, valid = self.viz.filter_points(uvz, (10, 10))
        self.assertEqual(int(mask.sum()), 2)
        self.assertEqual(int(valid.sum()), 3)


if __name__ == "__main__":
    unittest.main()
