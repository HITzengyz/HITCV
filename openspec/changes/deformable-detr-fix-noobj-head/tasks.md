## 1. Classification Head and Loss Alignment

- [x] 1.1 Change classification head output to `num_classes + 1` and define no-object index
- [x] 1.2 Update criterion to use `empty_weight` with `eos_coef` and correct logits shape
- [x] 1.3 Align matcher classification cost with new logits (ignore no-object)

## 2. Post-process and Debug

- [x] 2.1 Update post-processing to exclude no-object class in results export
- [x] 2.2 Add eval-time debug output for `pred_logits` shape and `p_noobj` (gated or once-per-eval)

## 3. Acceptance Verification

- [x] 3.1 Run eval to confirm `pred_logits` shape (B,300,8) and `noobj_idx=7`
- [x] 3.2 Confirm `p_noobj > 0.5` and `cardinality_error_unscaled` drops from ~297
- [x] 3.3 Re-run eval on same checkpoint/config to show non-zero AR/AP
