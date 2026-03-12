## Context

The current Deformable DETR pipeline assumes camera-style inputs only and has no radar-specific preprocessing, feature extraction, or fusion. This change introduces a radar pathway and fusion points while keeping existing vision-only behavior intact. The work is cross-cutting (data pipeline, model, config, evaluation) and affects training and inference flows.

## Goals / Non-Goals

**Goals:**
- Add a radar input pathway with preprocessing, encoder features, and fusion with existing vision features.
- Keep the radar pathway optional and backwards-compatible with vision-only runs.
- Add configuration surface and evaluation outputs for radar-assisted detection.

**Non-Goals:**
- Replacing existing vision encoders or changing core Deformable DETR attention math.
- Introducing new datasets or labeling standards beyond radar tensors and calibration metadata.
- Optimizing for specific hardware accelerators in this change.

## Decisions

1) **Radar feature encoder as a parallel branch**
   - **Decision:** Add a dedicated radar encoder that outputs features aligned to the vision feature pyramid levels.
   - **Alternatives:** (a) Project radar into image space and reuse vision encoder; (b) Fuse raw radar before any encoding.
   - **Rationale:** Parallel branch preserves radar-specific characteristics and avoids forcing radar into image-domain assumptions while enabling clean fusion at feature-level.

2) **Fusion at encoder/decoder attention inputs**
   - **Decision:** Concatenate or gated-sum radar features with vision features per pyramid level before attention.
   - **Alternatives:** (a) Late fusion on decoder outputs; (b) Cross-attention only for radar.
   - **Rationale:** Early/mid fusion allows attention to leverage radar signals in spatial refinement while limiting architectural disruption.

3) **Config-driven optionality**
   - **Decision:** Add config flags to enable radar input, specify radar preprocessing, and choose fusion mode.
   - **Alternatives:** Hard-code a single fusion approach.
   - **Rationale:** Keeps backward compatibility and supports ablations without code changes.

4) **Evaluation metrics extended, not replaced**
   - **Decision:** Add radar-assisted metrics alongside existing metrics.
   - **Alternatives:** Replace default metrics with radar-aware metrics.
   - **Rationale:** Preserves comparability with existing baselines.

## Risks / Trade-offs

- [Risk] Misaligned radar/vision calibration leads to degraded performance.  Mitigation: Validate calibration ingestion and add sanity checks in data loader.
- [Risk] Fusion increases compute/memory.  Mitigation: Make radar branch optional; allow lightweight encoder and configurable fusion.
- [Risk] Training instability from multi-sensor noise.  Mitigation: Provide normalization and dropout options for radar features.

## Migration Plan

- Add schema/config defaults that keep radar disabled by default.
- Roll out in three steps: data pipeline support, model branch + fusion, evaluation metrics.
- If issues arise, rollback by disabling radar flags and removing radar inputs from configs.

## Open Questions

- Exact radar tensor format and expected preprocessing steps (range-Doppler vs. point-cloud-like grids).
- Preferred fusion operator (concat + linear vs. gated sum) for baseline.
- Minimum evaluation metrics required for radar-assisted reporting.
