## Context

Current evidence shows transformed samples use resized `target["size"]` while `target["orig_size"]` remains sensor-native (e.g., 1080x1920). If postprocess scales prediction boxes with `orig_size` while GT/debug checks are in `size`, IoU collapses even when model outputs are otherwise reasonable. This can mask real learning and force AP to 0.

## Goals / Non-Goals

**Goals:**
- Make eval scaling source deterministic and dataset-aware for waterscenes.
- Preserve clear, inspectable logs that prove which scale source is used per eval run.
- Add strict category-domain checks to fail early on label-domain mismatch.
- Provide minimal diagnostics (10 overlays + IoU summary) to verify improvements.

**Non-Goals:**
- Implement category remapping in evaluator.
- Redesign training architecture or matcher logic.

## Decisions

1) **Scale-source selection**
   - Use `target["size"]` when `dataset.use_size_for_eval=True` (waterscenes), otherwise keep existing `orig_size` behavior.
   - Rationale: this is explicit, minimally invasive, and robust to dataset-specific resize pipelines.

2) **Dataset flag propagation**
   - Keep `use_size_for_eval=True` in waterscenes dataset build path and consume it in `evaluate()`.
   - Rationale: avoids overloading `--dataset_file` heuristics and keeps decision at data-construction boundary.

3) **Safety checks and observability**
   - Add eval checks for sample-vs-target size consistency and log scale source once per run.
   - Add category-domain hard check between predicted labels and GT category set.

4) **Diagnostics**
   - Reuse/add a minimal tool that saves 10 GT+Pred overlays and prints max IoU distributions (`max_any`, `max_same`).

## Risks / Trade-offs

- **Risk:** Hard category-domain check may stop runs that previously “worked” silently.
  - **Mitigation:** Error includes explicit label ranges and GT set.
- **Risk:** Different scale-source behavior across datasets.
  - **Mitigation:** gate through explicit dataset flag and log chosen source.

## Migration Plan

1) Enable dataset flag for waterscenes and scale-source selection in eval.
2) Add checks/logs and diagnostics outputs.
3) Run diagnostics and eval on same data/checkpoint before/after.
4) Accept only if IoU and AP/AR criteria improve and results.json sanity passes.

## Open Questions

- Should category-domain check support sparse-but-valid subsets (e.g., eval subset missing some classes) as warning instead of hard error?
