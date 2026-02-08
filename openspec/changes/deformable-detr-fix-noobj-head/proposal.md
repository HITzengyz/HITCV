## Why

Training is stuck with AP/AR=0 because the classification head outputs only `num_classes` logits and lacks the no-object class. This breaks background prediction, forces nearly all queries to be foreground, and collapses cardinality.

## What Changes

- Expand classification head output to `num_classes + 1` with no-object as the final class.
- Use a no-object weight (`eos_coef`) with the correct length in criterion.
- Update post-processing to ignore the no-object class when exporting results.
- Add eval-time debug print for `pred_logits` shape and no-object rate.

## Capabilities

### New Capabilities
- (none)

### Modified Capabilities
- `detection`: Classification head/output semantics and evaluation behavior for no-object handling.

## Impact

- Model heads and loss configuration (`deformable_detr.py`)
- Matching and post-processing (`matcher.py`, `deformable_detr.py` PostProcess)
- Eval logging (`engine.py`)
