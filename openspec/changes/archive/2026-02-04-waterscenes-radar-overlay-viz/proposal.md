## Why
- Radar integration can fail silently when projection, calibration, or augmentation sync is wrong; a point overlay is the fastest, most reliable way to visually validate alignment.
- We already support RGB(+TIR) early-fusion; a radar overlay is a prerequisite to safely produce REVP-style maps and expand to multi-modal training.
- We are about to add radar modality, so we need a hard visual gate now to avoid wasting training runs on misaligned data.

## What Changes
- Add a small CLI/script entry (e.g., `radar-overlay-viz`) that takes a frame id and outputs an overlay image, with an optional short video output.
- Support selectable radar attribute coloring for visualization: range, velocity/doppler, power/rcs (elevation optional if present).
- Add a simple multi-frame accumulation option (default 1 frame; optional 3-frame accumulation) to increase point density.
- Handle FOV mismatch by dropping points outside image bounds and optionally logging the dropped ratio.
- Non-goals: no radar-based detection, no training pipeline changes, no new model behavior in this change.

## Capabilities

### New Capabilities
- `radar-overlay-viz`: Project radar points into the image plane and render a debug overlay for calibration/augmentation verification.

### Modified Capabilities
- (none)

## Impact
- Dataset/IO: read radar point data and calibration (intrinsics/extrinsics), pair radar frames to camera frames/time without assuming new fields.
- Transforms pipeline: ensure overlay uses the same resize/crop/flip/pad as the training pipeline, or explicitly run pre-transform for raw calibration checks with an optional post-transform mode.
- Dependencies: likely OpenCV and/or matplotlib for rendering; must run on Windows dev environments; expectations about file layout and artifact output directories.

## Risks & Mitigations
- Timestamp mismatch or wrong pairing causes overlay drift.
  - Mitigation: log selected frame ids/timestamps and nearest-neighbor pairing deltas; allow an explicit override for pairing strategy.
- Axis convention or units mismatch (degrees vs radians, meters vs millimeters) causes systematic offset.
  - Mitigation: sanity-check by projecting a small known subset and verifying expected scale/orientation; add a simple config toggle for common convention flips.
- Transform sync issues (flip/resize/crop) make overlay "almost right" but still wrong.
  - Mitigation: provide both pre-transform and post-transform overlay outputs, with logs indicating which path was used.

## Acceptance / Done Criteria
- Given 3 representative sample frames, the tool outputs overlay PNGs with qualitatively correct alignment.
- Logs include counts for total radar points, in-FOV points, and out-of-FOV points per frame.
- At least 1 known-good frame is manually validated and stored under an `artifacts/` path (or a referenced path) for future smoke checks.
- Debug-only change: no training accuracy changes, and missing radar files fail gracefully with a clear error message (no crash).