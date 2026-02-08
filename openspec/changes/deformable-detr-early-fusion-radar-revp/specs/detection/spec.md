# Delta Spec: Deformable DETR Early-Fusion Radar + WaterScenes N-Modality Contract (REVP v1)

## Summary of Change
Introduce Radar as a minimal auxiliary modality for WaterScenes early-fusion, and add a WaterScenes-local modality dict + fusion contract to avoid hard-coded channel assumptions.

This delta is scoped to WaterScenes path only and does not refactor all datasets.

## Current Behavior
- Early-fusion input is fixed to 5 channels: `[rgb(3), tir(1), tir_valid(1)]`.
- Channel assumptions are hard-coded in model/backbone path.
- Radar is not consumed.
- `instances_train.json` may not contain `images[].tir_file`, so TIR depends on path derivation.

## Requirements

### Requirement: Strict TIR Preference
The system MUST prefer real TIR whenever a same-frame TIR image is resolvable under WaterScenes rules.

- RGB-only fallback (`tir=zeros`, `tir_valid=0`) is allowed ONLY when TIR is truly missing (no resolvable file).
- If a TIR path is resolved and file exists, loader MUST attempt to use real TIR.
- Strict behavior is controlled by `--tir_strict`:
  - `--tir_strict=1` (default): resolved-existing TIR read/decode failure MUST fail-fast (raise and stop), not silently fallback.
  - `--tir_strict=0`: resolved-existing TIR read/decode failure MAY fallback to zeros + `tir_valid=0` with rate-limited warning.

#### Scenario: TIR truly missing
- Given no `tir_file` and no resolvable `CAM_IR/<stem>.<ext>`
- When sample is loaded
- Then loader returns `tir=zeros`, `tir_valid=0`.

#### Scenario: TIR resolvable and readable
- Given `tir_file` or CAM_IR-derived path resolves to an existing readable file
- When sample is loaded
- Then loader returns real TIR and `tir_valid=1`.

#### Scenario: Strict mode read failure
- Given `--tir_strict=1` and TIR path resolves to an existing file but decode fails
- When sample is loaded
- Then loader raises a clear error and aborts (fail-fast).

#### Scenario: Non-strict mode read failure
- Given `--tir_strict=0` and TIR path resolves to an existing file but decode fails
- When sample is loaded
- Then loader falls back to zeros + `tir_valid=0` and emits a rate-limited warning with `repr(e)` and path.

### Requirement: CAM_IR Resolver Contract
Implement deterministic resolver:
- `resolve_tir_path(rgb_file_name, coco_root, tir_file=None) -> Optional[path]`

Resolver precedence:
1. If `tir_file` is present, accept absolute or dataset-root-relative `tir_file`.
2. Else derive from RGB stem using `CAM_IR/<stem>` with extension probing.
3. If none found, return `None`.

Extension probing order (case-insensitive):
- `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`

Stem rule:
- Stem is derived from RGB `file_name` by removing extension.
- Coverage tooling MUST report zero-padding mismatches (for example `00001` vs `1`) when they prevent overlap.

#### Scenario: `tir_file` authoritative
- Given valid `tir_file`
- When resolver runs
- Then `tir_file` result is used before CAM_IR derivation.

#### Scenario: CAM_IR fallback
- Given missing/invalid `tir_file`
- When resolver probes CAM_IR by stem and extension order
- Then first existing candidate path is returned.

### Requirement: TIR Read and Normalization Contract
TIR reader MUST support 8-bit and 16-bit grayscale payloads.

Normalization to float32 `[0,1]`:
- uint8 -> `/255.0`
- uint16 -> `/65535.0`
- float input -> clip `[0,1]`

### Requirement: Coverage and Observability
A coverage tool MUST be provided to quantify TIR availability and decode health.

`check_tir_coverage.py` output MUST include:
- `total_rgb`
- `total_tir`
- `overlap_count`
- `overlap_ratio`
- list of "resolvable but unreadable" TIR samples
- list of stem mismatch hints (including zero-padding candidates)

Training/inference observability MUST satisfy:
- `tir_valid_ratio` should approximately track measured `overlap_ratio` (small tolerance allowed).
- If `overlap_ratio > 0`, `tir_valid_ratio` MUST NOT stay `0`.

### Requirement: Missing-Modality Semantics (Strict-Aware)
- Truly missing TIR is treated as missing modality (`tir_valid=0`).
- Read/decode failure for an existing resolved TIR is NOT treated as missing in strict mode; it is an error.
- In non-strict mode only, read/decode failure may degrade to missing with warning.

## Existing Radar/Fusion Requirements (unchanged)
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

## Acceptance Tests
- Coverage script prints `total_rgb`, `total_tir`, `overlap_count`, `overlap_ratio`.
- Coverage script prints resolvable-but-unreadable list and mismatch hints.
- Strict mode (`--tir_strict=1`): any resolved-existing unreadable TIR triggers explicit failure.
- Non-strict mode (`--tir_strict=0`): unreadable TIR falls back with warning.
- 200-iteration run with `--tir_dropout 0`:
  - `tir_valid_ratio` approximately equals measured `overlap_ratio`
  - if `overlap_ratio > 0`, `tir_valid_ratio` is not `0`
  - if `overlap_ratio > 0.8`, `tir_valid_ratio > 0.8`
- Missing-modality semantics preserved: only truly-missing TIR yields default `tir_valid=0` in strict mode.

## WaterScenes Path Examples
- RGB: `model_examples/Deformable-DETR/data/waterscenes-coco/image/00001.jpg`
- TIR: `model_examples/Deformable-DETR/data/waterscenes-coco/CAM_IR/00001.jpg`
- Calibration: `model_examples/Deformable-DETR/data/waterscenes-coco/calib/00001.txt`
- Radar CSV pattern: `model_examples/Deformable-DETR/data/waterscenes-coco/radar/*.csv`
