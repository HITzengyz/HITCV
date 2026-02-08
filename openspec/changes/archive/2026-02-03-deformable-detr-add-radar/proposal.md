## Why

Current Deformable DETR support targets camera-style inputs only, leaving radar data unused in detection pipelines. Adding a radar pathway now enables better performance in adverse conditions and aligns with near-term multi-sensor requirements.

## What Changes

- Add a radar input pathway for Deformable DETR, including preprocessing and feature extraction for radar tensors.
- Introduce fusion of radar features with existing vision features during encoder/decoder attention.
- Extend data loading and configuration to accept radar frames and calibration metadata.
- Update evaluation outputs to report radar-assisted detection metrics alongside existing metrics.

## Capabilities

### New Capabilities
- `radar-input-fusion`: Accept radar inputs and fuse radar features with existing Deformable DETR feature maps for detection.

### Modified Capabilities
- (none)

## Impact

- Model architecture: new radar encoder branch and fusion points in attention blocks.
- Data pipeline: loaders, transforms, and batching for radar tensors and calibration.
- Config and CLI: new flags and config fields for radar inputs and fusion settings.
- Evaluation: additional metrics for radar-assisted detections.
