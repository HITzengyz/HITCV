## Context

Current radar fusion ingests a 1-channel radar map built from CSV u/v and power, with no explicit calibration-based projection in the training pipeline. The overlay tool already performs calibration-aware projection and shows that CSV u/v align with calib (sub-pixel error). Calibration files are per-frame text files containing 	_camera_radar (4x4 radar->camera) and 	_camera_intrinsic (3x4); extrinsics appear constant across frames, while intrinsics vary (max abs diff ~17 in sampled frames). The CSV includes x/y/z, range, doppler, azimuth, elevation, power, and u/v.

A direct spherical->(R,t)->K projection using azimuth/elevation yields very large u/v errors (deg and rad both poor). This suggests azimuth/elevation are not suitable for projection in this coordinate convention; the x/y/z columns are consistent with u/v and calibration.

## Goals / Non-Goals

**Goals:**
- Generate calibration-projected REVP radar tensors aligned to RGB resolution in the dataset pipeline.
- Keep the existing feature-level fusion (RadarEncoder + pre-transformer fusion) unchanged, only expanding radar input channels.
- Support missing radar data by emitting zero tensors and radar_valid=0 without hard failure.
- Use per-frame calibration for intrinsics (K) while keeping R/t consistent with calibration files.

**Non-Goals:**
- New fusion architectures (cross-attn, BEV fusion, geometric alignment modules).
- Distortion modeling or image undistortion.
- Motion compensation or temporal radar accumulation.
- Changes to model outputs or evaluation protocols beyond radar-assisted metrics already present.

## Decisions

1) **REVP v0 channel schema**
- Use 4 channels: range, doppler, elevation, power, plus radar_valid mask (total 5 radar channels). This matches available CSV fields and aligns with minimal change constraints.

2) **Projection source data**
- Use CSV x/y/z for projection into the image plane via radar->camera extrinsics and camera intrinsics.
- Rationale: u/v computed from x/y/z and calib match the CSV u/v at ~0.26 px mean error, while azimuth/elevation-based spherical projection yields large errors.

3) **Calibration strategy**
- Load per-frame calibration files (matching frame id) to obtain K and R/t.
- Rationale: sampled extrinsics are effectively constant, but intrinsics vary across frames (max abs diff ~17). Per-frame loading is correct and low complexity.

4) **Aggregation policy**
- For multi-point collisions at a pixel, select the point with maximum power and copy its range/doppler/elevation/power values into REVP channels.
- Rationale: preserves strongest return and mirrors radar salience; simplest deterministic rule.

5) **Normalization defaults (conservative)**
- range: clip [0, 200] then /200
- doppler: clip [-20, 20] then (x+20)/40
- elevation: clip [-10, 10] (deg) then (x+10)/20
- power: clip [0, P99] then /P99 (P99 from dataset stats)

## Risks / Trade-offs

- **Azimuth/elevation unit ambiguity** ¡ú Mitigation: avoid using azimuth/elevation for projection; validate elevation unit for normalization using dataset stats and document assumption (degrees).
- **Per-frame calib file missing/mismatched** ¡ú Mitigation: fall back to radar_valid=0 with zero tensors and log a warning.
- **Intrinsics variability** ¡ú Mitigation: load per-frame K, avoid static intrinsics.
- **Power-max aggregation may suppress weaker points** ¡ú Mitigation: revisit aggregation if model underutilizes radar; consider mean or weighted schemes in later iterations.
