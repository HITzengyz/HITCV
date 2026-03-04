## 1. Dataset + Calibration Projection

- [x] 1.1 Update model_examples/Deformable-DETR/detr/datasets/coco.py get_radar to build REVP (range/doppler/elevation/power) from CSV using per-frame calib and x/y/z projection
- [x] 1.2 Allow missing radar files in CocoDetection.__getitem__ by emitting zero tensors and radar_valid=0 instead of raising
- [x] 1.3 Add per-frame calib loading (match frame id to calib/<id>.txt) and reuse t_camera_radar/t_camera_intrinsic

## 2. Transforms + Normalization

- [x] 2.1 Expand datasets/transforms.py Normalize concat logic to append 4 radar channels + radar_valid (total radar channels=5)
- [x] 2.2 Update ModalityDropout to zero all radar channels and set radar_valid=0
- [x] 2.3 Update make_coco_transforms mean/std defaults to include REVP channels

## 3. Model Radar Pathway

- [x] 3.1 Update model_examples/Deformable-DETR/detr/models/deformable_detr.py RadarEncoder(in_channels=5)
- [x] 3.2 Adjust radar tensor slicing to pull 5 radar channels from input
- [x] 3.3 Update any radar pipeline sanity tests (detr/test/check_radar_pipeline.py) to reflect channel count

## 4. Validation + Smoke Tests

- [x] 4.1 Add/adjust a small script or test to verify REVP tensor shapes and radar_valid behavior (missing radar case)
- [x] 4.2 Run overlay stats check (in-FOV ratio) using tools/radar_overlay_viz.py on a small batch
- [x] 4.3 Run 1-epoch smoke training with radar enabled to confirm end-to-end stability
