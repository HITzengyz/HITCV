## Why

AP=0 persists with near-zero prediction-vs-GT IoU, and evidence points to a coordinate-system mismatch in evaluation scaling (`size` vs `orig_size`). A fix limited to eval is insufficient unless training-side/debug-side coordinate consistency and label-domain checks are enforced end-to-end.

## What Changes

- Enforce a stable target-size source for postprocessing scale in evaluation, with explicit dataset-gated behavior for waterscenes.
- Propagate waterscenes dataset flag (`use_size_for_eval=True`) from dataset build to evaluation logic.
- Add mandatory eval sanity checks and warnings for size consistency and chosen scale source.
- Add minimal diagnostics that save 10 GT+Pred overlays and print IoU distribution stats.
- Add strict category-domain consistency checks in eval (hard fail on mismatch, no remapping).

## Capabilities

### New Capabilities
- (none)

### Modified Capabilities
- `detection`: Coordinate-system consistency for prediction scaling and evaluator-domain sanity checks.

## Impact

- Eval pipeline and postprocess scaling behavior: `detr/engine.py`
- Waterscenes dataset flags and propagation: `detr/datasets/coco.py`
- Diagnostic tooling and IoU stats: `detr/tools/*`
- Eval safety checks for category domain and result sanity.
