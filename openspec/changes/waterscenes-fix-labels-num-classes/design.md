## Context

Waterscenes training uses COCO-style annotations where `category_id` is 1-based and not guaranteed contiguous. The current Deformable-DETR code path reads `category_id` directly and defaults to COCO class count (91) when `dataset_file='coco'`, which misaligns labels and class dimensions. The change must fix label indexing while keeping the original JSON intact, and allow explicit class-count configuration so matcher/loss/model heads stay consistent.

## Goals / Non-Goals

**Goals:**
- Provide a safe, repeatable path to generate a 0-based contiguous label JSON from the waterscenes annotations without overwriting originals.
- Ensure training can set `num_classes=7` explicitly and that all class-dependent components use the same value.
- Keep the fix minimal and compatible with existing training scripts.

**Non-Goals:**
- Redesign dataset schemas or re-encode images.
- Change evaluation logic beyond class-count alignment.
- Introduce new annotation formats.

## Decisions

1) **Offline JSON normalization vs. loader-time remap**
   - **Decision:** Generate a new normalized JSON file (e.g., `instances_train_0based.json`) and point training to it.
   - **Why:** Keeps the loader simple, avoids hidden label remaps, and preserves the original dataset for audit/rollback.
   - **Alternatives:** Runtime mapping inside `CocoDetection` (rejected to avoid silent label shifts and harder debugging).

2) **Explicit `--num_classes` override**
   - **Decision:** Add `--num_classes` CLI arg that, when set, overrides the default class count in model construction and loss/matcher config.
   - **Why:** Works for any non-COCO dataset size (7 here), avoids coupling `dataset_file` to class count, and keeps compatibility with existing configs.
   - **Alternatives:** Add dataset-specific `dataset_file` branch for waterscenes (rejected as brittle and not extensible).

3) **Contiguous class IDs (0..C-1) requirement**
   - **Decision:** Enforce 0-based contiguous IDs in normalized JSON; rely on `--num_classes` to match C.
   - **Why:** Matcher and focal loss expect a dense class range; avoids indexing errors and label skew.

## Risks / Trade-offs

- **Risk:** Users forget to switch to the normalized JSON → **Mitigation:** Provide clear documentation and a validation check to warn if label min/max not in [0, C-1].
- **Risk:** `--num_classes` not passed during training → **Mitigation:** Default behavior unchanged; recommend training command updates and optionally log computed class stats.
- **Risk:** Multiple JSONs for train/val drift → **Mitigation:** Normalize both train and val in the same run and store side by side.

## Migration Plan

1) Generate new 0-based contiguous JSONs for train/val without overwriting originals.
2) Update training command to point `--coco_path` to folder containing normalized JSONs (or direct annotation path if supported).
3) Pass `--num_classes 7` to training.
4) Run a short overfit sanity check (1–5 images) to validate loss decrease and non-empty predictions.
5) Rollback by switching back to original JSONs and removing `--num_classes` if needed.

## Open Questions

- Do we want to auto-detect class count from the JSON when `--num_classes` is absent?
- Should we add a lightweight warning if labels are not contiguous or not 0-based?
