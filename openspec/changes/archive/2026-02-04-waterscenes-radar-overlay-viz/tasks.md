## 1. Repo Recon & Entrypoints
- [x] 1.1 Locate WaterScenes COCO dataset directory and confirm RGB image root path (record in notes) (waterscenes-coco layout uses `<root>/image` in `model_examples/Deformable-DETR/detr/datasets/coco.py`; default root is `model_examples/Deformable-DETR/data/waterscenes-coco`)

## 2. Repo Recon & Entrypoints
- [x] 2.1 Locate dataset loader entrypoint(s) used for WaterScenes (record file paths) (`model_examples/Deformable-DETR/detr/main.py` -> `datasets/build_dataset` -> `model_examples/Deformable-DETR/detr/datasets/__init__.py` -> `datasets/coco.py`)

## 3. Repo Recon & Entrypoints
- [x] 3.1 Locate transforms/augmentations implementation (record file paths and relevant functions) (`model_examples/Deformable-DETR/detr/datasets/coco.py:make_coco_transforms` and `model_examples/Deformable-DETR/detr/datasets/transforms.py`)

## 4. Repo Recon & Entrypoints
- [x] 4.1 Identify best location for a debug CLI/script entrypoint (e.g., tools/ or scripts/) (`model_examples/Deformable-DETR/detr/tools/` exists; `model_examples/Deformable-DETR/detr/test/` holds other debug scripts)

## 5. Repo Recon & Entrypoints
- [x] 5.1 Enumerate existing calibration/parsing utilities (if any) and note reuse opportunities (`datasets/coco.py` loads `radar_calib` from `--radar_calib` as identity or npy/txt; `CocoDetection.get_radar` parses CSV and builds a power map; transforms in `datasets/transforms.py` already handle radar/tir channels)

## 6. Repo Recon & Entrypoints
- [x] 6.1 Confirm output/artifacts folder conventions (record preferred path pattern) (existing outputs under `model_examples/Deformable-DETR/detr/test/output` and `model_examples/Deformable-DETR/detr/result`; use `model_examples/Deformable-DETR/detr/result/<frame_id>_overlay.png` for quick runs and repo-root `artifacts/` for manual review samples per task 55)

## 7. Repo Recon & Entrypoints
- [x] 7.1 Confirm whether RGB+TIR early-fusion path exists and whether it affects overlay placement (TIR is loaded in `datasets/coco.py` via `tir_root` and merged in `datasets/transforms.py` with radar; overlay must follow the same transform pipeline to stay aligned)

## 8. Calibration & IO (Default Path Required)
- [x] 8.1 Implement calibration loader for intrinsics K from default path: model_examples\Deformable-DETR\data\waterscenes-coco\calib (implemented in `model_examples/Deformable-DETR/detr/tools/radar_overlay_viz.py`)

## 9. Calibration & IO (Default Path Required)
- [x] 9.1 Implement calibration loader for radar->camera extrinsics (R,t) from default path: model_examples\Deformable-DETR\data\waterscenes-coco\calib (implemented in `model_examples/Deformable-DETR/detr/tools/radar_overlay_viz.py`)

## 10. Calibration & IO (Default Path Required)
- [x] 10.1 Add a CLI arg `--calib_dir` to override default calibration folder

## 11. Calibration & IO (Default Path Required)
- [x] 11.1 Add error handling: missing calib files => log + mark calib_invalid (no crash)

## 12. Calibration & IO (Default Path Required)
- [x] 12.1 Add validation for K shape and R/t dimensions; log validation failures

## 13. Calibration & IO (Default Path Required)
- [x] 13.1 Document calibration filename expectations in script help output

## 14. Radar Data Loading & Time Sync
- [x] 14.1 Implement radar reader for per-frame point attributes (range/azimuth/elevation/doppler/power/rcs) and timestamps if available

## 15. Radar Data Loading & Time Sync
- [x] 15.1 Implement nearest-neighbor timestamp association with default `--max_delta 0.05s`

## 16. Radar Data Loading & Time Sync
- [x] 16.1 Log selected radar frame id and timestamp delta per RGB frame

## 17. Radar Data Loading & Time Sync
- [x] 17.1 Handle missing radar files gracefully (log + produce RGB-only overlay)

## 18. Radar Data Loading & Time Sync
- [x] 18.1 Handle empty radar point sets (log + produce RGB-only overlay)

## 19. Radar Data Loading & Time Sync
- [x] 19.1 Add optional per-frame metadata output indicating radar_valid flag

## 20. Radar Data Loading & Time Sync
- [x] 20.1 Add unit conversion toggles for azimuth/elevation (deg/rad) via CLI config

## 21. Projection Pipeline & Stats
- [x] 21.1 Implement spherical->Cartesian conversion in radar frame (range/azimuth/elevation)

## 22. Projection Pipeline & Stats
- [x] 22.1 Implement radar->camera extrinsic transform p_cam = R * p_radar + t

## 23. Projection Pipeline & Stats
- [x] 23.1 Implement intrinsic projection with K to pixel coordinates (u,v)

## 24. Projection Pipeline & Stats
- [x] 24.1 Filter invalid points: NaNs, z<=0, and out-of-bounds pixels

## 25. Projection Pipeline & Stats
- [x] 25.1 Compute and log stats: total points, in-FOV points, out-of-FOV points, dropped ratio

## 26. Projection Pipeline & Stats
- [x] 26.1 Implement collision policy for REVP map (power/rcs max wins) and log policy used

## 27. Projection Pipeline & Stats
- [x] 27.1 Add optional point ordering by power/rcs for overlay drawing

## 28. Projection Pipeline & Stats
- [x] 28.1 Validate projection output ranges and log anomalies (e.g., extreme u/v)

## 29. Rendering & Outputs
- [x] 29.1 Implement point overlay rendering on RGB (default point radius 2 px)

## 30. Rendering & Outputs
- [x] 30.1 Implement selectable color mode: range

## 31. Rendering & Outputs
- [x] 31.1 Implement selectable color mode: doppler/velocity

## 32. Rendering & Outputs
- [x] 32.1 Implement selectable color mode: power/rcs

## 33. Rendering & Outputs
- [x] 33.1 Implement optional REVP rasterization (H x W x 4) and alpha-blend overlay

## 34. Rendering & Outputs
- [x] 34.1 Define output naming convention: <out_dir>/<frame_id>_overlay.png

## 35. Rendering & Outputs
- [x] 35.1 Implement per-frame JSON stats output (counts, dropped ratio, mode)

## 36. Rendering & Outputs
- [x] 36.1 Implement optional short video output from ordered overlays

## 37. Rendering & Outputs
- [x] 37.1 Implement batch mode for folder/list of frames with continue-on-error

## 38. Rendering & Outputs
- [x] 38.1 Add summary report after batch run (success/fail counts, mean dropped ratio)

## 39. Rendering & Outputs
- [x] 39.1 Ensure outputs are deterministic for identical inputs and config

## 40. Transform-Sync Verification Mode
- [x] 40.1 Implement raw (pre-transform) overlay mode for calibration-only checks

## 41. Transform-Sync Verification Mode
- [x] 41.1 Implement transform-synced overlay mode using same resize/crop/flip/pad params as training pipeline

## 42. Transform-Sync Verification Mode
- [x] 42.1 Provide explicit mode selection flag and log which mode is produced

## 43. Transform-Sync Verification Mode
- [x] 43.1 Verify both raw and synced overlays can be emitted for the same frame in one run

## 44. Transform-Sync Verification Mode
- [x] 44.1 Add sanity check to compare raw vs synced overlay sizes and log mismatches

## 45. CLI / Entrypoint
- [x] 45.1 Define CLI arguments (frame id, out_dir, render mode, max_delta, calib_dir, batch list)

## 46. CLI / Entrypoint
- [x] 46.1 Add `--max_delta` default 0.05s and `--point_radius` default 2 px

## 47. CLI / Entrypoint
- [x] 47.1 Add `--accumulate_k` optional (default 1) with warning if >1 and no motion compensation

## 48. CLI / Entrypoint
- [x] 48.1 Add `--stats_json` toggle and default output path behavior

## 49. CLI / Entrypoint
- [x] 49.1 Add `--continue_on_error` toggle for batch processing

## 50. Tests & Proof
- [x] 50.1 Unit test: missing radar file -> overlay produced, radar_valid=0, no crash

## 51. Tests & Proof
- [x] 51.1 Unit test: empty radar points -> overlay produced, counts zero, no crash

## 52. Tests & Proof
- [x] 52.1 Unit test: missing calibration -> clear error + skip frame, no crash

## 53. Tests & Proof
- [x] 53.1 Unit test: horizontal flip keeps projected points consistent with image flip

## 54. Tests & Proof
- [x] 54.1 Unit test: projection filters NaNs and z<=0 correctly

## 55. Tests & Proof
- [x] 55.1 Produce >=20 overlay images and store under artifacts/ for manual review (generated under `model_examples/Deformable-DETR/detr/visualization/`; earlier overlays in `model_examples/Deformable-DETR/detr/artifacts/waterscenes-radar-overlay-viz/`)

## 56. Tests & Proof
- [x] 56.1 Validate at least one known-good frame and record its path in a README or notes (see `openspec/changes/waterscenes-radar-overlay-viz/notes.md`)

## 57. Tests & Proof
- [x] 57.1 Run batch overlay command on 10 samples and save summary log (counts + timestamp deltas) (log at `model_examples/Deformable-DETR/detr/visualization/batch_10_summary.log`)
