# Design: Deformable DETR Early-Fusion RGB+TIR

## Overview
Implement early-fusion by expanding input to 5 channels and wiring TIR through dataset and transforms. The backbone conv1 in_channels changes from 3 to 5. No changes to Transformer/Head structure.

## Data Flow
1) Dataset reads RGB via `file_name` and optional TIR via `tir_file`.
2) Spatial transforms are applied to both RGB and TIR.
3) Convert to tensors; create tir_valid.
4) Concatenate channels to form [5, H, W].
5) Backbone conv1 accepts 5 channels.

## Dataset Changes
- Extend COCO dataset loader to read `tir_file` (if present).
- If `tir_file` is missing/unreadable:
  - TIR channel is all zeros.
  - tir_valid channel is all zeros.
- If `tir_file` is present:
  - Load as single-channel (grayscale) image.
  - tir_valid is 1.

## Transform Changes
- Spatial transforms (resize/crop/flip/pad) should apply to TIR consistently.
- Normalization must accept 5 channels:
  - Use explicit mean/std for 5 channels: RGB(3), TIR(1), tir_valid(1).
  - tir_valid can be normalized with mean=0, std=1 to preserve 0/1 values.

## Modality Dropout
- Add `ModalityDropout(p)` applied only during training.
- Behavior: when dropped, set TIR to zeros and tir_valid to 0.
- Default p=0.3; configurable via CLI/args.

## Logging/Validation
- Add log or visualization step to confirm input tensor channel count == 5.
- Add shape assertion in forward input or dataset check.

## Testing Plan
- Unit/script check for:
  - input shape [5, H, W]
  - value ranges are stable after normalization
  - tir_valid correctness for present/missing TIR
- Smoke test: 1 epoch training.
- Inference test with missing TIR: model runs without crash.

## Backward Compatibility
- Existing RGB-only datasets continue to work (tir_file missing). The model should run with TIR zero-filled and tir_valid=0.
