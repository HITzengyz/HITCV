## Why

`model_examples/Deformable-DETR` currently mixes core model code, dataset-specific fusion logic, debugging tools, and large generated artifacts. This makes onboarding slow and increases the risk of editing non-source or non-critical files.

## What Changes

- Add a newcomer-oriented architecture guide for `model_examples/Deformable-DETR/detr`.
- Document the runtime call path from `main.py` to model forward/loss/eval.
- Classify repository content into:
  - core model-critical code
  - training/data/eval support code
  - non-core or generated artifacts that should not be treated as source of truth
- Add a practical reading order for users unfamiliar with Deformable-DETR.
- Add explicit "safe-to-edit vs avoid-editing-first" boundaries.

## Capabilities

### New Capabilities
- `codebase-orientation`: A maintained, file-grounded orientation contract for Deformable-DETR code understanding and navigation.

### Modified Capabilities
- (none)

## Impact

- New docs in `model_examples/Deformable-DETR/detr/docs/` and/or related README pointers.
- No model behavior, training logic, or inference outputs are changed.
- Reduces onboarding ambiguity and accidental edits to generated artifacts.
