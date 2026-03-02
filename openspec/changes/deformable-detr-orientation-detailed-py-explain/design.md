## Context

The recently added `codebase-orientation` capability gives a high-level map, but newcomers still need deeper explanations for each Python file listed in section 2 (core + runtime support). The immediate gap is not only "what this file is" but also "how data/tensors move through it" with evidence-based shape notes.

Key constraints:
- This is documentation-only work.
- Shape annotations must be source-grounded; if a dimension cannot be justified from code, it must be marked unknown.
- Existing orientation structure should remain navigable for beginners.

## Goals / Non-Goals

**Goals:**
- Define a consistent per-file explanation template for all core/support Python files listed in section 2.
- Include module responsibilities, key classes/functions, call relationships, and typical input/output contracts.
- Add tensor-shape propagation notes where verifiable from code and explicitly mark unknowns where not inferable.
- Keep the document beginner-friendly and cross-linked to source paths.

**Non-Goals:**
- Modify model/training/evaluation code behavior.
- Add benchmark claims or shape values not inferable from code.
- Cover every utility/test script in the repository; scope remains section-2 core/support files.

## Decisions

1) **Per-file explanation schema**
- For each Python file, use a stable structure:
  - Role in pipeline
  - Main symbols (classes/functions)
  - Callers/callees in runtime path
  - Data contract (inputs/outputs)
  - Shape notes (if inferable)
- Rationale: uniform structure improves scanability and teaching value.

2) **Shape evidence policy**
- Annotate shapes only when backed by code assertions, fixed operators, or explicit construction logic.
- When shape depends on runtime config/data, use symbolic form (`B`, `C`, `H`, `W`, `Nq`, etc.) and state dependency source.
- If unknown, write "unknown from static inspection".
- Rationale: avoids misleading readers with fabricated certainty.

3) **Keep summary + deep detail layered**
- Preserve section-2 table as quick map.
- Add detailed per-file subsections after the table.
- Rationale: maintain fast orientation while enabling deeper study.

4) **Doc-only acceptance checks**
- Require file existence verification for all referenced paths.
- Require each listed core/support Python file to have a corresponding subsection.
- Rationale: prevents drift and partial coverage.

## Risks / Trade-offs

- **Risk:** Document becomes too long for first-time readers.  
  **Mitigation:** keep section-2 table concise and move depth into collapsible/logical subsections.

- **Risk:** Shape notes may still be misread as guaranteed runtime values.  
  **Mitigation:** label symbolic and configuration-dependent dimensions explicitly.

- **Risk:** Future code changes can stale parts of explanations.  
  **Mitigation:** add a lightweight maintenance checklist (file path checks + key symbol checks).

## Migration Plan

1. Add/refresh section-2 detailed subsections in `codebase_orientation.md`.
2. Validate coverage against current section-2 core/support Python file list.
3. Validate all linked file paths exist.
4. Review for shape-evidence policy compliance (no unsupported hard numbers).

## Open Questions

- Should detailed explanations remain in one file, or be split into companion docs per module group (models/datasets/engine/util)?
- Should shape notes include one optional concrete example configuration (clearly marked as example) or remain purely symbolic?
