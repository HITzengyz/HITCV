# Proposal: Deformable DETR Early-Fusion Radar + WaterScenes N-Modality Contract (REVP v1)

## Summary
Add a follow-up, minimal-diff enhancement to Deformable DETR early-fusion for WaterScenes only:
- Keep existing RGB+TIR behavior intact.
- Add Radar as an auxiliary modality using a minimal REVP raster representation.
- Introduce a WaterScenes-local modality dict + fusion step so channel composition is configuration-driven instead of hard-coded.

This change is intentionally scoped to avoid broad refactors and preserve existing non-WaterScenes dataset behavior.

## Why This Change
The current input path is fixed to 5 channels (RGB+TIR+tir_valid). This blocks safe extension to Radar and increases silent-failure risk as modalities evolve.

WaterScenes already provides per-frame calibration and radar measurements:
- Calibration: `model_examples/Deformable-DETR/data/waterscenes-coco/calib/<frame>.txt`
- Radar points: `model_examples/Deformable-DETR/data/waterscenes-coco/radar/<frame>.csv`

A minimal radar projection + raster map enables immediate multimodal experimentation with low implementation risk.

## Goals
- Add Radar in early-fusion with minimal representation: `k<=4` channels in v1.
- Use v1 radar channels: `occupancy`, `range`, `doppler`, `power` (`k=4`).
- Add `radar_valid` channel and consistent missing-modality semantics.
- Introduce modality dict + fusion order contract only in WaterScenes path.
- Keep modality dropout train-only.
- Add deterministic transform-alignment and missing-modality robustness tests.

## Non-Goals
- No broad refactor across all datasets.
- No redesign of DETR transformer/head/loss.
- No high-dimensional radar feature engineering (e.g., 13-channel representation) in v1.
- No change to existing TIR change artifacts or behavior outside this follow-up scope.

## Scope Boundaries
- Apply modality dict + fusion transform only in WaterScenes code path (`detr/datasets/coco.py`, `detr/datasets/transforms.py`).
- Keep other dataset layouts and pipelines unaffected.

## Risks Addressed
- Silent read failures for TIR/Radar/calib producing ambiguous behavior.
- Geometric misalignment across RGB/TIR/Radar transforms.
- Validity-channel corruption through normalization/casting.
- Hard-coded input channels preventing safe modality growth.

## Acceptance Criteria
- WaterScenes fused input contract implemented: `[rgb(3), tir(1), tir_valid(1), radar_k(4), radar_valid(1)]` -> `in_channels = 10`.
- Missing modalities never crash and always produce zero channels + valid=0.
- Deterministic transform test proves exact pixel alignment across RGB/TIR/Radar maps.
- Normalization sanity checks pass (valid channels remain {0,1}; mean/std length matches fused channel count).
- 1-epoch smoke train completes with logs showing fused channel count and modality valid ratios.
