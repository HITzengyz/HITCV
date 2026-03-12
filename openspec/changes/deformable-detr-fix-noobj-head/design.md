## Context

Current training shows `pred_logits` shaped `(B, 300, 7)` with no-object index 6 and `p_noobj=0.0`, which collapses background prediction and yields `cardinality~297` with AP/AR=0. The classification head lacks the no-object class, and downstream criterion/post-processing logic assumes a no-object class exists.

## Goals / Non-Goals

**Goals:**
- Make the classification head output `num_classes + 1` logits with no-object as the last class.
- Ensure criterion uses an `empty_weight` of length `num_classes + 1` with `eos_coef` applied to the no-object class.
- Ensure post-processing ignores the no-object class when exporting results.
- Add eval-time debug output for `pred_logits` shape and `p_noobj` (optionally gated).

**Non-Goals:**
- Change label normalization or dataset semantics.
- Redesign matcher strategy beyond aligning with no-object handling.

## Decisions

1) **Switch to explicit no-object class in logits**
   - **Decision:** `class_embed` outputs `num_classes + 1`; no-object index is `num_classes`.
   - **Why:** Aligns with DETR-style background handling and fixes degenerate foreground collapse.
   - **Alternative:** Keep `num_classes` logits and treat background implicitly (rejected due to observed collapse).

2) **Criterion uses cross-entropy with `empty_weight`**
   - **Decision:** Use `F.cross_entropy` with `empty_weight` length `num_classes + 1`, and set `empty_weight[-1] = eos_coef`.
   - **Why:** Explicitly weights no-object and restores correct background training signal.

3) **Post-process excludes no-object**
   - **Decision:** Use `softmax` over logits and drop the last class before top-k selection.
   - **Why:** Prevents exporting no-object detections.

4) **Eval-time debug logging**
   - **Decision:** Print `pred_logits` shape and `p_noobj` once per eval (optionally behind a flag).
   - **Why:** Provides direct sanity check for background handling without heavy instrumentation.

## Risks / Trade-offs

- **Risk:** Switching loss from focal to CE may change convergence behavior → **Mitigation:** Keep `eos_coef` configurable and validate on small overfit run.
- **Risk:** Post-process change may affect existing evaluation baselines → **Mitigation:** Limit to excluding no-object class and keep box conversion unchanged.
- **Risk:** Debug logs add noise → **Mitigation:** Print once per eval or behind a flag.

## Migration Plan

1) Update model head output dimension and criterion loss.
2) Update matcher and post-processing to align with new logits.
3) Add eval-time debug print.
4) Re-run eval on the same checkpoint/config and confirm non-zero AP/AR and improved cardinality.

## Open Questions

- Should debug printing be gated by a new CLI flag or always-on for eval?
