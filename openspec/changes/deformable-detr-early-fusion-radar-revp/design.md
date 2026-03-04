# Design: Deformable DETR Early-Fusion Radar + WaterScenes N-Modality Contract (REVP v1)

## 1. Design Overview
This design adds Radar as an auxiliary modality while keeping implementation minimal and localized.

Key decisions:
1) Introduce modality dict + fusion in WaterScenes path only.
2) Keep early-fusion backbone strategy unchanged (single concatenated tensor).
3) Use a small radar raster (`k=4`) plus `radar_valid` channel.
4) Add strict TIR preference so existing resolvable TIR is never silently ignored.

## 2. Grounded Diagnosis (2026-02-08)
Observed on current local dataset:
- `instances_train.json` contains no `images[].tir_file` keys.
- WaterScenes RGB root is `image/` and TIR root is `CAM_IR/`.
- Current loader fallback is effectively `CAM_IR/<file_name>` with no extension probing.
- Local file coverage is sparse: 100 TIR files for 33,500 RGB files (~0.299%).

Implication:
- Low `tir_valid_ratio` may be true data sparsity.
- But resolver and strict semantics are still required to prevent silent modal collapse when TIR exists but decode path is broken.

## 3. Data Sources and Paths
For WaterScenes frame `<stem>` (for example `00001`):
- RGB: `image/<stem>.jpg` (from COCO `file_name`)
- TIR: `CAM_IR/<stem>.<ext>` (optional, extension may vary)
- Radar: `radar/<stem>.csv` (optional)
- Calibration: `calib/<stem>.txt` (optional)

## 4. Resolver Design
Implement deterministic resolver:
- `resolve_tir_path(rgb_file_name, coco_root, tir_file=None) -> Optional[path]`

Resolution precedence:
1) `tir_file` if present and valid:
- absolute path
- root-relative path (`coco_root / tir_file`)
2) Derived CAM_IR fallback by RGB stem:
- parse stem from `rgb_file_name`
- probe case-insensitive extensions in order:
  - `.png`
  - `.jpg`
  - `.jpeg`
  - `.tif`
  - `.tiff`
3) If nothing exists, return `None`.

Design note:
- Resolver stays deterministic and does not auto-normalize stem values.
- Coverage tooling reports stem mismatch diagnostics, including zero-padding issues (`00001` vs `1`).

## 5. Strict TIR Preference State Machine
```text
resolve_tir_path(...) -> tir_path?
        |
        +-- None -------------------------------> truly missing -> zeros + tir_valid=0
        |
        +-- path exists -> try decode/normalize
                          |
                          +-- success ----------> use real TIR + tir_valid=1
                          |
                          +-- failure
                               |
                               +-- tir_strict=1 -> raise (fail-fast)
                               |
                               +-- tir_strict=0 -> zeros + tir_valid=0 + warning
```

Behavioral contract:
- RGB-only degradation is allowed only for truly missing TIR.
- Existing-resolved TIR decode failure is an error in strict mode.

## 6. TIR Read/Decode Policy
Reader requirements:
- accept 8-bit and 16-bit grayscale payloads
- output float32 single-channel in `[0,1]`

Normalization:
- uint8 `/255.0`
- uint16 `/65535.0`
- float inputs clipped to `[0,1]`

Failure payload:
- include attempted path
- include `repr(e)`
- warning rate-limited once per epoch per worker (non-strict only)

## 7. Missing-Modality Semantics (Strict-Aware)
- Truly missing TIR: valid missing modality (`tir_valid=0`).
- Resolved-existing TIR decode failure:
  - strict mode: error, not missing fallback
  - non-strict mode: fallback to missing semantics with warning

This replaces older wording that treated all read failures as missing.

## 8. WaterScenes-Local N-Modality Contract
### 8.1 Dataset Output Contract (WaterScenes path only)
Dataset produces modality entries:
- `rgb`
- `tir`
- `tir_valid`
- `radar_k`
- `radar_valid`

### 8.2 Fusion Contract
Fusion order:
- `[rgb(3), tir(1), tir_valid(1), radar_k(4), radar_valid(1)]`

`in_channels = 10` for REVP v1.

## 9. Coverage and Observability Design
Add `detr/tools/check_tir_coverage.py` with outputs:
- `total_rgb`
- `total_tir`
- `overlap_count`
- `overlap_ratio`
- resolvable-but-unreadable list
- mismatch hints (including zero-padding candidates)

Training/inference logs:
- `tir_valid_ratio`
- `radar_valid_ratio`
- compare `tir_valid_ratio` against measured `overlap_ratio`
- `overlap_ratio > 0` should never co-exist with `tir_valid_ratio == 0`

## 10. Debug Path (Optional, Not Default)
Phase-A debugging may enable paired-only sampling mode:
- train only on overlap subset where TIR is resolvable/readable
- purpose: isolate and prove TIR pipeline correctness when global overlap is very low
- this is debug-only and must not become default data policy

## 11. Radar/Transform/Normalization (Unchanged)
- Radar projection and raster policy unchanged.
- Transform alignment must remain deterministic across RGB/TIR/Radar.
- Valid channels remain identity-normalized.

## 12. Verification Plan Additions
1) Coverage script validates overlap and decode health.
2) Strict mode fail-fast tests for unreadable but existing TIR.
3) Non-strict fallback tests preserve warning semantics.
4) 200-iteration run (`--tir_dropout 0`) compares `tir_valid_ratio` to measured overlap.
5) Optional paired-only debug run demonstrates TIR path activity clearly.
