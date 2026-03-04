## 1. Data & Input Pipeline

- [x] 1.1 Define radar tensor format and required calibration metadata schema
- [x] 1.2 Add radar data loading, validation, and error handling for missing calibration
- [x] 1.3 Implement radar preprocessing pipeline and normalization hooks
- [x] 1.4 Integrate radar batching with vision samples in dataloader

## 2. Model & Fusion

- [x] 2.1 Implement radar encoder branch producing pyramid-aligned feature maps
- [x] 2.2 Add fusion operator(s) for radar + vision features before attention
- [x] 2.3 Wire radar features into encoder/decoder attention path when enabled
- [x] 2.4 Ensure vision-only path unchanged when radar is disabled

## 3. Configuration & CLI

- [x] 3.1 Add config fields for radar enablement, preprocessing, and fusion mode
- [x] 3.2 Add configuration validation and defaults (radar disabled by default)
- [x] 3.3 Expose CLI or config overrides for radar settings

## 4. Evaluation & Metrics

- [x] 4.1 Define radar-assisted metric outputs and report format
- [x] 4.2 Implement metric collection for radar-enabled evaluation runs
- [x] 4.3 Ensure evaluation reports include standard and radar-assisted metrics

## 5. Tests & Documentation

- [x] 5.1 Add unit tests for radar input validation and preprocessing
- [x] 5.2 Add integration test covering radar-enabled forward pass with fusion
- [x] 5.3 Update docs or README for radar input requirements and config usage
