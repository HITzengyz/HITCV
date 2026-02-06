# Tasks

- [x] 1) Create WaterScenes-local modality dict contract in dataset
  - Files: `model_examples/Deformable-DETR/detr/datasets/coco.py`
  - Targets:
    - `CocoDetection.__getitem__`
    - WaterScenes path resolution logic
  - Work:
    - Build modality entries for `rgb`, `tir`, `tir_valid`, `radar_k`, `radar_valid` in WaterScenes path only.
    - Keep non-WaterScenes behavior unchanged.
  - Acceptance:
    - WaterScenes sample exposes required modality keys before fusion.

- [x] 2) Add robust missing-modality handling + rate-limited warnings
  - Files: `model_examples/Deformable-DETR/detr/datasets/coco.py`, `model_examples/Deformable-DETR/detr/main.py`
  - Targets:
    - `CocoDetection.__getitem__`
    - dataset epoch hook (e.g., `set_epoch`)
    - train loop epoch boundary hook
  - Work:
    - TIR read failure -> zeros + `tir_valid=0` with warning once per epoch per worker.
    - Radar/calib read failure -> zeros + `radar_valid=0` with warning once per epoch per worker.
  - Acceptance:
    - Repeated failures in one epoch emit at most one warning per modality per worker.

- [x] 3) Implement Radar projection + REVP v1 raster (`k=4`)
  - Files: `model_examples/Deformable-DETR/detr/datasets/coco.py` (or helper module under datasets)
  - Targets:
    - radar CSV loader
    - calib parser for `t_camera_radar` and `t_camera_intrinsic`
    - projection/raster functions
  - Work:
    - Parse radar CSV columns and calib matrices.
    - Project points to image plane with `z>0` and bounds filtering.
    - Rasterize channels: `occupancy`, `range`, `doppler`, `power`.
    - Collision policy: occupancy OR, range min, doppler mean, power max.
  - Acceptance:
    - For valid radar+calib sample, produced `radar_k` has shape `[4,H,W]` and finite values.

- [x] 4) Add WaterScenes-local fusion transform order and computed in_channels
  - Files: `model_examples/Deformable-DETR/detr/datasets/transforms.py`, `model_examples/Deformable-DETR/detr/datasets/coco.py`, `model_examples/Deformable-DETR/detr/main.py`
  - Targets:
    - normalize/fusion transform stage
    - args/config for modality order and radar channels
  - Work:
    - Fuse in order `[rgb, tir, tir_valid, radar_k, radar_valid]`.
    - Compute `in_channels = 5 + k + 1` (`k=4` default).
    - Pass expected channels to model/backbone construction.
  - Acceptance:
    - Fused tensor shape matches expected in_channels on train/val WaterScenes.

- [x] 5) Generalize channel assertions with minimal diff
  - Files: `model_examples/Deformable-DETR/detr/models/deformable_detr.py`, `model_examples/Deformable-DETR/detr/models/backbone.py`
  - Targets:
    - input-channel assertion in model forward
    - backbone conv1 initialization channel count
  - Work:
    - Replace hard-coded `5` with config-driven `in_channels`.
  - Acceptance:
    - Model starts successfully with in_channels=10 for radar v1.

- [x] 6) Add train-only Radar modality dropout
  - Files: `model_examples/Deformable-DETR/detr/datasets/transforms.py`, `model_examples/Deformable-DETR/detr/datasets/coco.py`, `model_examples/Deformable-DETR/detr/main.py`
  - Targets:
    - transform pipeline for train split only
  - Work:
    - Add Radar dropout parameter and transform.
    - Ensure eval/test pipelines contain no random modality dropout.
  - Acceptance:
    - Train can zero radar modality stochastically; val/test deterministic.

- [x] 7) Deterministic transform alignment unit test (RGB/TIR/Radar)
  - Files: `model_examples/Deformable-DETR/detr/test/` (new test script)
  - Targets:
    - deterministic resize/crop/flip/pad case
  - Work:
    - Build synthetic sample with known pixel markers in all modalities.
    - Verify exact mapped pixel coordinates across modalities after transform.
  - Acceptance:
    - Test fails on mismatch, passes on exact correspondence.

- [x] 8) Missing-modality inference tests
  - Files: `model_examples/Deformable-DETR/detr/test/` (extend/add scripts)
  - Cases:
    - no-TIR
    - no-radar
    - both missing
  - Checks:
    - No crash
    - `tir_valid`/`radar_valid` maps correct (all zeros when missing)
  - Acceptance:
    - All three cases pass in CI/local test script.

- [x] 9) Normalization and validity-channel sanity checks
  - Files: `model_examples/Deformable-DETR/detr/test/` (extend/add checks)
  - Work:
    - Assert mean/std length equals fused channel count.
    - Assert `tir_valid` and `radar_valid` remain in `{0,1}` after fusion and dtype conversions (fp16/bf16 where available).
  - Acceptance:
    - Sanity script exits successfully; violations produce explicit assertion errors.

- [ ] 10) 1-epoch smoke training + evidence logging
  - Files: train script/log config under `model_examples/Deformable-DETR/detr/test/` and runtime logging points in dataset/model paths
  - Work:
    - Run 1 epoch on WaterScenes.
    - Capture logs with:
      - fused channel count
      - `tir_valid_ratio`
      - `radar_valid_ratio`
  - Acceptance:
    - Smoke run completes and logs contain required evidence fields.


