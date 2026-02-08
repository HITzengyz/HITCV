## ADDED Requirements

### Requirement: inputs-and-schema
#### Scenario: accept-generic-sensor-inputs
- **WHEN** the tool is provided an RGB frame, a radar point set with generic attributes (range, azimuth, elevation, doppler, power/rcs), camera intrinsics, radar-to-camera extrinsics, and timestamps for both sensors.
- **THEN** the system MUST accept these inputs without requiring dataset-specific field names and MUST treat all missing optional attributes as absent rather than error.

### Requirement: temporal-association
#### Scenario: nearest-neighbor-pairing
- **WHEN** a radar frame timestamp is within `max_delta` of the RGB timestamp (default `0.05s`).
- **THEN** the system MUST associate the nearest radar frame to the RGB frame; if no radar frame is within `max_delta`, the system MUST mark radar as missing for that RGB frame.

### Requirement: projection-geometry
#### Scenario: filter-invalid-projections
- **WHEN** projecting radar points into the camera image plane using intrinsics/extrinsics.
- **THEN** the system MUST drop points with NaNs, non-positive depth (z <= 0), or pixel coordinates outside the image bounds and MUST report the in-FOV vs out-of-FOV counts.

### Requirement: output-artifacts
#### Scenario: overlay-and-stats-output
- **WHEN** an overlay is generated for a frame.
- **THEN** the system MUST write an overlay image to an explicit output path/name and SHOULD emit a per-frame JSON stats record (counts, dropped ratio, selected render mode).

### Requirement: rendering-modes
#### Scenario: selectable-color-encoding
- **WHEN** the user selects a render attribute mode.
- **THEN** the system MUST support at least `range`, `doppler/velocity`, and `power/rcs` modes and MUST render points with a configurable point radius (default `2 px`).

### Requirement: robustness
#### Scenario: missing-or-empty-radar
- **WHEN** radar data is missing, empty, or calibration is unavailable for a frame.
- **THEN** the system MUST NOT crash, MUST log a clear warning, and MUST still output an RGB image (either unchanged or with zero points) in a deterministic manner.

### Requirement: batch-processing
#### Scenario: process-sequence-continue-on-error
- **WHEN** processing a directory or sequence of N frames.
- **THEN** the system MUST continue on per-frame errors, record failures, and produce outputs for remaining frames.

### Requirement: augmentation-verification
#### Scenario: raw-vs-transformed-overlay
- **WHEN** augmentation verification mode is enabled.
- **THEN** the system MUST be able to emit both a raw (pre-transform) overlay and a transform-synced overlay, or clearly indicate which mode was produced.