# Delta Spec: Deformable DETR Early-Fusion Radar + WaterScenes N-Modality Contract (REVP v1)

## Summary of Change
Introduce Radar as a minimal auxiliary modality for WaterScenes early-fusion, and add a WaterScenes-local modality dict + fusion contract to avoid hard-coded channel assumptions.

This delta is scoped to WaterScenes path only and does not refactor all datasets.

## Current Behavior
- Early-fusion input is fixed to 5 channels: `[rgb(3), tir(1), tir_valid(1)]`.
- Channel assumptions are hard-coded in model/backbone path.
- Radar is not consumed.

## New Behavior
### WaterScenes intermediate modality contract (v1 only)
Dataset path for WaterScenes returns/propagates modality entries:
- `rgb`
- `tir`
- `tir_valid`
- `radar_k`
- `radar_valid`

### Fused tensor contract (WaterScenes)
Fusion order is explicit:
- `[rgb(3), tir(1), tir_valid(1), radar_k(k), radar_valid(1)]`

For REVP v1:
- `k = 4` (`occupancy`, `range`, `doppler`, `power`)
- `in_channels = 5 + k + 1 = 10`

## Radar Projection + Raster Contract
Inputs:
- Radar CSV: `radar/<stem>.csv`
- Calib TXT: `calib/<stem>.txt` with `t_camera_radar` (4x4), `t_camera_intrinsic` (3x3)

Projection:
- `X_cam = T_camera_radar * X_radar`
- `u,v = (K * X_cam) / z`, keep `z>0` and in-bounds

Raster output:
- shape `[k,H,W]`, `k=4` in v1
- collision policy:
  - occupancy: OR
  - range: min
  - doppler: mean
  - power: max

## Missing-Modality Semantics
Mandatory behavior:
- Missing/failed TIR: `tir=zeros`, `tir_valid=0`
- Missing/failed Radar or calib: `radar_k=zeros`, `radar_valid=0`

No random tensors are allowed as fallback for missing/failed modalities.

## Dropout Requirements
- Modality dropout is train split only.
- Eval/test must not random-drop modalities.
- Radar dropout mirrors TIR dropout semantics (`radar_k=zeros`, `radar_valid=0` when dropped).

## Transform Consistency Requirements
Spatial transforms must be identical across RGB/TIR/Radar maps using one sampled transform outcome per operation:
- resize
- crop
- horizontal flip
- pad

## Normalization Requirements
- Mean/std list length must equal fused channel count.
- Valid channels (`tir_valid`, `radar_valid`) are identity-normalized (mean=0, std=1).
- Valid channels must remain binary-equivalent `{0,1}` after fusion and dtype conversions.

## Logging Requirements
Training/runtime logs must include:
- fused channel count
- `tir_valid_ratio`
- `radar_valid_ratio`
- rate-limited read-failure warnings for TIR/Radar/calib

## Acceptance Tests
- No-TIR inference: no crash, correct `tir_valid` map.
- No-Radar inference: no crash, correct `radar_valid` map.
- Both missing inference: no crash, both valid maps correct.
- Deterministic transform alignment test: exact pixel mapping across RGB/TIR/Radar.
- Normalization sanity checks:
  - valid channels remain in `{0,1}`
  - mean/std length equals fused channel count.
- 1-epoch smoke train with logs containing fused channel count and valid ratios.

## WaterScenes Path Examples
- `model_examples\Deformable-DETR\data\waterscenes-coco\calib\00001.txt`
- `model_examples\Deformable-DETR\data\waterscenes-coco\radar\*.csv`
