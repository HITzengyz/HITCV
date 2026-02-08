# Proposal: Deformable DETR Early-Fusion RGB+TIR

## Summary
Enable DrivingSDK Deformable DETR to accept early-fusion 5-channel inputs: RGB(3) + TIR(1) + tir_valid(1). The system must handle missing TIR robustly by filling TIR with zeros and setting tir_valid to 0.

## Goals
- Support 5-channel early-fusion input end-to-end.
- Keep DETR/Transformer/Head structure unchanged.
- Add configurable training-time Modality Dropout (TIR only, default p=0.3).
- Extend COCO JSON images with optional field `tir_file` pointing to the TIR image path.
- Ensure missing TIR does not crash inference.

## Non-Goals
- No changes to Transformer, Head, or loss structures beyond required input handling.
- No changes to annotation formats beyond optional `tir_file`.
- No new model architectures or fusion strategies beyond early-fusion.

## Constraints
- Only minimal changes: dataset input stitching, backbone conv1 in_channels, normalization/augmentation consistency, optional modality dropout.
- When TIR is missing: TIR channel = 0, tir_valid = 0.

## Risks
- Input normalization must remain stable for 5-channel input.
- Augmentations must be applied consistently to RGB and TIR.
- Any mismatch in H/W between RGB and TIR must be handled or detected.

## Acceptance Criteria
- Forward pass asserts input tensor shape == [5, H, W].
- 1 epoch smoke test completes.
- Missing TIR inference test passes without crash.
- Logs or visualization prove 5-channel inputs are used.
