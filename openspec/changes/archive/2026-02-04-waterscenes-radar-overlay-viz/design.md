## Context
- WaterScenes uses monocular RGB with 4D radar, plus GPS/IMU; the overlay is a debug visualization that projects radar points into the RGB image plane to validate calibration + transform sync.
- Current pipeline loads images (RGB, and in some paths RGB+TIR) and applies torchvision-style transforms before feeding COCO-style targets; the viz should hook into the same dataset/transforms layer without changing training outputs.
- Constraints: radar FOV exceeds camera FOV, radar is sparse/noisy, timestamps may be offset, and calibration may be missing or incomplete for some samples.

## Goals / Non-Goals
**Goals:**
- Given one sample frame, load radar points, project to image plane, and generate:
  - a) point overlay image (RGB with colored points)
  - b) optional REVP dense map (H x W x 4) and alpha-blended overlay
- Ensure transforms sync: resize/crop/flip/pad applied identically to RGB and radar projection outputs.
- Provide a CLI/script or notebook entrypoint to batch-generate overlays for N samples for sanity check.
- Robustness: handle empty radar, out-of-FOV points, or missing calibration without hard crash (log + skip).
**Non-Goals:**
- Training a detector with radar features in this change.
- Building BEV features or any BEV/PV fusion network.
- Perfect occlusion reasoning (no z-buffer unless trivial and optional).

## Decisions
### D1. Visualization Representation
- Decision: Default to direct point overlay; optional REVP rasterization as a secondary output.
- Rationale: Point overlay is the smallest reliable debug tool; REVP is useful for early-fusion later but not required for v0 correctness.
- Alternatives: REVP-only outputs (heavier, more assumptions about rasterization and channel semantics).

### D2. Pixel Collision Policy
- Decision: Max power/rcs wins per pixel for REVP; for point overlay, draw all points but prefer last-write by power order.
- Rationale: Max power is stable and preserves strongest reflector; mean can blur sparse reflectors.
- Alternatives: Mean aggregation or nearest-depth wins.

### D3. Multi-frame Accumulation
- Decision: v0 uses 1-frame only; v1 adds K-frame accumulation (K=3) without ego-motion compensation.
- Rationale: Keeps v0 minimal and avoids incorrect motion compensation assumptions; K-frame improves density for debugging but is optional.
- Alternatives: Add ego-motion compensation using GPS/IMU (postponed until calibration and timing are verified).

### D4. Calibration & Projection
- Decision: Implement explicit radar->camera projection with well-defined frames and strict validity checks.
- Rationale: Avoids silent errors and makes axis/units mismatches visible in logs.
- Key formulas / edge cases:
  - Radar spherical to Cartesian (radar frame):
    - x = r * cos(elev) * cos(az)
    - y = r * cos(elev) * sin(az)
    - z = r * sin(elev)
  - Transform to camera frame: p_cam = R * p_radar + t
  - Project to pixels: [u v 1]^T ~ K * p_cam, with z > 0
  - Drop invalid points: NaNs, z <= 0, u/v outside image bounds.

### D5. Transform Sync Strategy
- Decision: Strategy 1 (shared sample dict + transforms operate on keys: image, revp, points).
- Rationale: Single source of truth for transform parameters; lowest risk for drift between RGB and radar outputs.
- Alternatives: Strategy 2 (transform RGB, then apply equivalent affine to points and re-rasterize) is more error-prone and duplicates logic.

### D6. Integration Points
- Decision: Keep dataset __getitem__ returning raw image + radar points + calibration (when available); add a separate viz entrypoint that uses the dataset + shared transforms to produce overlays.
- Rationale: Minimal disruption to training; overlay tool is debug-only and can share the same IO stack.
- Alternatives: Inject overlay generation into dataset returns (would couple debug outputs to training and slow iteration).

## Risks / Trade-offs
- [FOV mismatch] -> Drop points outside image bounds; log in-FOV vs out-of-FOV counts.
- [Timestamp misalignment] -> Use nearest-neighbor pairing; log delta; warn if delta exceeds threshold.
- [Sparsity] -> Optional K-frame accumulation; report density stats; keep point overlay as fallback.
- [Augmentations break alignment] -> Enforce shared-transform pipeline; add unit tests for flip/crop consistency.
- [Performance] -> Vectorize projection; optional batched drawing with OpenCV.

## Acceptance Tests / Proof
- Run a command that produces overlay images for 10 samples into an output dir.
- At least one screenshot/saved image manually verified.
- Unit test: flipping RGB also flips projected points consistently.
- Unit test: empty radar produces a valid image with 0 points and no crash.