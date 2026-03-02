## Context

The current `Deformable-DETR` example contains both active source code and many run outputs (`visualization/`, `kernel_meta/`, `test/output*`, etc.). New contributors can confuse generated files or diagnostics with core model implementation, especially after multiple WaterScenes/TIR/Radar adaptations.

## Goals / Non-Goals

**Goals**
- Provide an accurate, file-level map of the actual runtime path.
- Clearly separate core model code from support utilities and generated artifacts.
- Provide a reading sequence for first-time users.
- Keep guidance grounded in the current repository tree.

**Non-Goals**
- No refactor of model/dataset/training code.
- No behavior changes to training, inference, or evaluation.
- No cleanup or deletion of generated artifacts in this change.

## Decisions

1) **Single source of orientation truth**
- Use `model_examples/Deformable-DETR/detr` as the analysis root.
- Derive structure from current files, not from upstream README assumptions.

2) **Three-layer classification**
- Layer A: core model mechanism files (transformer/attention/matcher/loss/backbone).
- Layer B: runtime support files (dataset pipeline, engine, eval wrappers, tools/scripts).
- Layer C: non-core generated artifacts/log outputs.

3) **Entry-point first explanation**
- Start from `main.py -> build_model -> forward/loss -> evaluate`.
- Then branch into dataset fusion and modality handling.

4) **Actionable boundaries**
- Add explicit "edit-first" and "do-not-edit-first" lists.

## Risks / Trade-offs

- **Risk:** docs become stale as code evolves.
  - **Mitigation:** include a compact validation checklist and keep file references concrete.
- **Risk:** oversimplification of custom WaterScenes adaptations.
  - **Mitigation:** document both original Deformable-DETR core and project-specific modifications side by side.

## Verification Plan

1. Confirm all referenced files exist.
2. Confirm the documented call path matches current imports/calls.
3. Confirm classification examples include at least one item from each layer.
