## Why

Radar fusion currently consumes a 1-channel u/v map without enforcing geometric projection from calibration, which can yield weak or pseudo-aligned radar signals. We already have calibration-aware overlay tooling; aligning training inputs with that projection via REVP will make radar information physically meaningful and more learnable.

## What Changes

- Generate calibration-projected REVP radar tensors (range/doppler/elevation/power + radar_valid) aligned to RGB resolution during dataset loading.
- Preserve existing feature-level fusion (RadarEncoder + pre-transformer fusion), but expand radar input channels to match REVP.
- Allow missing radar data by emitting zero tensors with radar_valid=0 instead of hard failure.
- Update normalization defaults to cover REVP channels.

## Capabilities

### New Capabilities
- (none)

### Modified Capabilities
- adar-input-fusion: Update radar preprocessing requirements to accept REVP multi-channel inputs and tolerate missing radar data with explicit validity masking.

## Impact

- Data loading and preprocessing: model_examples/Deformable-DETR/detr/datasets/coco.py, model_examples/Deformable-DETR/detr/datasets/transforms.py
- Model radar pathway: model_examples/Deformable-DETR/detr/models/deformable_detr.py
- Configuration defaults: model_examples/Deformable-DETR/detr/main.py
- Validation: overlay stats + smoke tests for radar present/missing
