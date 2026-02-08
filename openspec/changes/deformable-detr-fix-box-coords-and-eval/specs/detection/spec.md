## ADDED Requirements

### Requirement: Waterscenes eval SHALL use resized target size for box scaling
The system SHALL scale predicted boxes with `target["size"]` during evaluation for waterscenes datasets and SHALL NOT scale with fixed sensor-native `orig_size`.

#### Scenario: Waterscenes eval scaling
- **WHEN** evaluating a waterscenes sample and `use_size_for_eval=True`
- **THEN** `PostProcess` receives `target["size"]` as `target_sizes`

### Requirement: Eval SHALL log and validate size consistency
The system SHALL validate and log size conventions before accumulating evaluation outputs.

#### Scenario: Size consistency check
- **WHEN** eval processes a batch
- **THEN** it checks `samples(H,W)` against `target["size"]=[h,w]` and logs warning with `image_id`, sample size, target size, and orig_size if inconsistent

#### Scenario: Scale-source logging
- **WHEN** eval starts
- **THEN** it logs the effective scale source used for postprocess (`size` or `orig_size`)

### Requirement: Eval SHALL enforce category-domain consistency
The system SHALL validate predicted label domain against GT categories and fail fast on mismatch.

#### Scenario: Domain mismatch
- **WHEN** predicted labels include ids outside GT category domain
- **THEN** eval aborts with explicit mismatch details (pred min/max and GT category set summary)

### Requirement: Diagnostic outputs SHALL support coordinate alignment verification
The system SHALL provide minimal outputs to verify coordinate correctness.

#### Scenario: Overlay and IoU diagnostics
- **WHEN** diagnostic tool runs
- **THEN** it saves 10 GT+Pred overlay images and prints `max_any`/`max_same` IoU min/median/max

### Requirement: Results export SHALL stay COCO-compliant and sane
The system SHALL export bbox results in pixel `xywh` and surface bbox sanity stats.

#### Scenario: Results sanity
- **WHEN** `results.json` is produced
- **THEN** bbox entries are pixel `xywh`, and the system reports out-of-bounds and invalid-width/height rates
