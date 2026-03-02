## Why

Current orientation content identifies core/support files, but does not provide file-by-file explanations deep enough for newcomers to quickly understand responsibilities and execution flow. Readers also lack trustworthy tensor-shape guidance, which slows debugging and model comprehension.

## What Changes

- Expand the orientation scope to require detailed explanations for each Python file listed under "core" and "runtime support" in `codebase_orientation.md` section 2.
- Define documentation requirements for shape propagation notes at module boundaries and key forward paths.
- Require strict shape-reporting policy: only documented/inferable shapes are allowed; unknown dimensions must be explicitly marked unknown instead of guessed.
- Define acceptance criteria for readability and coverage for beginner readers.

## Capabilities

### New Capabilities
- (none)

### Modified Capabilities
- `codebase-orientation`: Extend requirements from high-level classification to per-file deep explanations with evidence-based shape annotations.

## Impact

- Documentation targets:
  - `model_examples/Deformable-DETR/detr/docs/codebase_orientation.md`
  - optionally companion docs under `model_examples/Deformable-DETR/detr/docs/`
- Validation/consistency checks for links and file references in the orientation docs.
- No changes to model behavior, training pipeline, evaluation metrics, or runtime outputs.
