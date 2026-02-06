# Design: Deformable DETR Early-Fusion Radar + WaterScenes N-Modality Contract (REVP v1)

## 1. Design Overview
This design adds Radar as an auxiliary modality while keeping implementation minimal and localized.

Key decisions:
1) Introduce modality dict + fusion in WaterScenes path only.
2) Keep early-fusion backbone strategy unchanged (single concatenated tensor).
3) Use a small radar raster (`k=4`) plus `radar_valid` channel.
4) Preserve existing missing-modality semantics and make them explicit for Radar.

## 2. Data Sources and Paths
For WaterScenes frame `<stem>` (e.g., `00001`):
- RGB: `image/<stem>.jpg`
- TIR: `CAM_IR/<stem>.jpg` (optional)
- Radar: `radar/<stem>.csv` (optional)
- Calibration: `calib/<stem>.txt` (optional)

Calibration file contains:
- `t_camera_radar`: 4x4 extrinsic matrix
- `t_camera_intrinsic`: 3x3 camera intrinsic matrix `K`

If explicit metadata is absent in COCO `images` entry, paths are derived by `file_name` stem.

## 3. WaterScenes-Local N-Modality Contract
### 3.1 Dataset Output Contract (WaterScenes path only)
Dataset produces modality entries (intermediate contract):
- `rgb`: RGB image/tensor
- `tir`: single-channel map (or missing)
- `tir_valid`: scalar/map validity
- `radar_k`: `k`-channel radar raster (`k=4` in v1)
- `radar_valid`: scalar/map validity

Other datasets continue existing behavior unchanged.

### 3.2 Fusion Contract
Fusion order (v1 fixed, config-backed):
- `[rgb(3), tir(1), tir_valid(1), radar_k(4), radar_valid(1)]`

Thus:
- `in_channels = 3 + 1 + 1 + 4 + 1 = 10`
- General formula retained: `in_channels = 5 + k + 1`

## 4. Radar Projection Math
Given radar point in radar frame (homogeneous):
- `X_radar = [x, y, z, 1]^T`

Project to camera frame:
- `X_cam = T_camera_radar * X_radar`

Perspective projection:
- `uv_h = K * [X_cam.x, X_cam.y, X_cam.z]^T`
- `u = uv_h.x / X_cam.z`
- `v = uv_h.y / X_cam.z`

Validity filters:
- Keep only points with `X_cam.z > 0`
- Keep only in-bounds pixels: `0 <= u < W`, `0 <= v < H`

## 5. Radar Rasterization Policy (REVP v1)
Raster grid aligned to RGB image size (`H x W`), channels `k=4`:
1) `occupancy`: binary 0/1 if any valid point falls into pixel.
2) `range`: minimum camera-depth (or Euclidean range) among points per pixel.
3) `doppler`: mean doppler per pixel.
4) `power`: max power per pixel.

Collision handling (multiple points to one pixel):
- `occupancy`: logical OR
- `range`: min aggregator
- `doppler`: running mean
- `power`: max aggregator

Pixels without points remain zero.

## 6. Missing-Modality Semantics
Mandatory, consistent behavior:
- Missing/failed TIR read: `tir=zeros`, `tir_valid=0`
- Missing/failed Radar read OR missing/failed calib parse: `radar_k=zeros`, `radar_valid=0`

No random fallback tensors. Failures are explicit via valid channels.

## 7. Dropout Policy
- Modality dropout applies to train split only.
- Eval/test never random-drop modalities.
- TIR dropout behavior retained.
- Add Radar dropout with analogous semantics:
  - when dropped: `radar_k=zeros`, `radar_valid=0`

## 8. Transform Consistency
All spatial transforms must use one sampled transform outcome and apply identically across RGB/TIR/Radar maps:
- resize
- crop
- horizontal flip
- pad

This is required to maintain geometric alignment for projected radar map supervision.

## 9. Normalization Policy
- Normalize fused channels with explicit mean/std list sized to fused channels.
- Valid channels (`tir_valid`, `radar_valid`) must remain identity-like (mean=0, std=1).
- Sanity assertions:
  - mean/std length == fused channel count
  - valid channels are binary-equivalent {0,1} after fusion/casting checks.

## 10. Logging and Observability
Add lightweight runtime logs:
- Fused channel count (once at model start)
- Train-epoch modality valid ratios:
  - `tir_valid_ratio`
  - `radar_valid_ratio`
- Read-failure warnings for TIR/Radar/calib with rate-limit (once per epoch per worker).

## 11. Compatibility and Minimal-Diff Strategy
- Existing TIR-only early-fusion behavior remains functionally unchanged unless WaterScenes radar features are enabled.
- Non-WaterScenes datasets continue current path.
- Backbone/forward channel assertions move from hard-coded `5` to computed `in_channels` from fusion config.

## 12. Testing Strategy Summary
1) Missing-modality inference robustness:
- no-TIR, no-radar, both-missing: no crash, valid channels correct.
2) Deterministic transform alignment test:
- exact pixel correspondence for RGB/TIR/Radar after known transform.
3) Normalization sanity:
- valid channels in {0,1}; mean/std length equals fused channels.
4) Smoke training:
- 1 epoch success with required logs.

## 13. WaterScenes Path Examples
- Calibration: `model_examples\Deformable-DETR\data\waterscenes-coco\calib\00001.txt`
- Radar CSV pattern: `model_examples\Deformable-DETR\data\waterscenes-coco\radar\*.csv`
