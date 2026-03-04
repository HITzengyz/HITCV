# Delta Spec: Deformable DETR Early-Fusion RGB+TIR

## Summary of Change
Add early-fusion input support for RGB+TIR with a 5-channel tensor: [R, G, B, TIR, tir_valid]. Keep the Deformable DETR model structure unchanged; only adjust input handling (dataset/transforms) and backbone conv1 in_channels.

## Current Behavior
- Input is RGB only: 3-channel tensor [3, H, W].
- COCO JSON uses standard `file_name` for RGB images.
- Missing modalities are not supported.

## New Behavior
- Input is 5-channel tensor [5, H, W]: RGB(3) + TIR(1) + tir_valid(1).
- COCO JSON `images` entries may include optional `tir_file` pointing to TIR image path.
- If `tir_file` is missing or unreadable, TIR is all zeros and tir_valid is 0.
- During training, optional Modality Dropout can zero only TIR and set tir_valid=0 (default p=0.3, configurable).

## Data Contract
### COCO JSON image entry
- Required: `file_name` (RGB)
- Optional: `tir_file` (TIR image path; grayscale image expected)

### Input Tensor
- Shape: [5, H, W]
- Channels:
  - 0..2: RGB normalized
  - 3: TIR normalized or zero
  - 4: tir_valid mask (0 or 1), broadcast to [1, H, W]

## Config/CLI Changes
- Add argument: `--tir_dropout` (float, default 0.3) to control modality dropout probability.
- Add argument: `--tir_norm_mean` and `--tir_norm_std` if required; otherwise reuse RGB stats with explicit 5-channel normalization.

## Transform/Normalization Requirements
- All spatial transforms (resize/crop/flip/pad) must be applied consistently to RGB and TIR.
- Normalization must support 5-channel input, including tir_valid channel.

## Acceptance Tests
- Forward pass asserts input tensor shape == [5, H, W].
- 1 epoch smoke test runs successfully.
- Missing TIR inference test passes without crash.
- Logs/visualizations demonstrate 5-channel inputs are consumed.
