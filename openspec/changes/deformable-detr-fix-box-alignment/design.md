## Context

Despite fixing no-object handling, AP/AR can remain near zero, suggesting a box alignment issue between transformed targets, model outputs, and evaluation sizes. We need minimal diagnostics to identify whether the problem is in transforms (xyxy/cxcywh normalization), size conventions (H,W), or post-processing scaling.

## Goals / Non-Goals

**Goals:**
- Produce concrete visualizations of transformed GT boxes and predictions on the same samples.
- Quantify alignment via max IoU distributions (same-class and class-agnostic).
- Log and validate size conventions across samples/targets/post-process.
- Apply the smallest fix in transforms or target sizing if misalignment is found.

**Non-Goals:**
- Change model architecture or training regime.
- Redefine dataset annotation formats beyond alignment fixes.

## Decisions

1) **Diagnostics-first workflow**
   - **Decision:** Add scripts to visualize GT-only and GT+pred overlays and compute IoU distributions before modifying alignment logic.
   - **Why:** Avoids guessing and ensures any fix is evidence-driven.

2) **Minimal fixes in transforms/target sizes**
   - **Decision:** Prefer adjusting target size propagation or box normalization/scaling only where misalignment is confirmed.
   - **Why:** Limits regression risk and keeps fixes localized.

3) **Consistent H/W logging**
   - **Decision:** Log `samples(H,W)`, `target["size"]`, and `target["orig_size"]` once per eval to confirm order `[h,w]`.
   - **Why:** Many silent failures stem from swapped dimensions.

## Risks / Trade-offs

- **Risk:** Visualizations slow down eval → **Mitigation:** Limit to 5 samples and write to a dedicated output folder.
- **Risk:** IoU script misinterprets class mapping → **Mitigation:** Provide both same-class and class-agnostic IoU modes.

## Migration Plan

1) Run diagnostics on current checkpoint and data subset.
2) Apply the minimal alignment fix based on evidence.
3) Re-run diagnostics and eval to confirm IoU > 0.3 appears and AP/AR > 0.

## Open Questions

- Is any class mapping still inconsistent between train and eval?
