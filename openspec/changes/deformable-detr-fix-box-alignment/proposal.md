## Why

AP remains 0 despite correct class/no-object handling, suggesting box/size alignment or target scaling issues. We need concrete diagnostics and minimal fixes to ensure transforms, target sizes, and post-processing agree on coordinate conventions.

## What Changes

- Add diagnostic tooling to visualize transformed GT boxes and predictions on the same samples.
- Add IoU distribution analysis (same-class and class-agnostic) to quantify alignment.
- Log and validate size/shape conventions (H,W) across samples, targets, and post-processing.
- Apply a minimal fix (if identified) in transforms/target sizing to restore alignment.

## Capabilities

### New Capabilities
- (none)

### Modified Capabilities
- `detection`: Box/size alignment diagnostics and corrected target size handling if misaligned.

## Impact

- Dataset transforms and target sizing (`datasets/transforms.py`, `datasets/coco.py`)
- Eval/post-processing (`models/deformable_detr.py`, `engine.py`)
- New diagnostics scripts/tools (visualization + IoU stats)
