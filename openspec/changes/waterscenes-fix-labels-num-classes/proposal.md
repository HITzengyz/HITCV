## Why

Training on waterscenes currently yields AP=0 because labels are 1-based and the model uses a mismatched class count (default COCO 91). This causes incorrect class supervision and prevents learning.

## What Changes

- Normalize waterscenes COCO annotations to 0-based contiguous category IDs (0..C-1) and write a new JSON without overwriting the original.
- Allow explicit `--num_classes` to override defaults so training/evaluation use the true class count (7) consistently in matcher/loss/model heads.

## Capabilities

### New Capabilities
- (none)

### Modified Capabilities
- `detection`: Class label normalization and explicit class-count configuration for training/evaluation.

## Impact

- Data preprocessing / annotation tooling (new JSON output alongside original).
- Training configuration and model build path to honor `--num_classes`.
- Matcher/loss behavior (class dimension consistency).
