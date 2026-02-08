# Spec: Radar Input Fusion

## Purpose
Define requirements for ingesting radar tensors and fusing radar features with vision features for Deformable DETR training and inference.

## Requirements

### Requirement: Accept radar input tensors
The system SHALL accept radar tensors and optional calibration metadata alongside existing vision inputs for Deformable DETR training and inference.

#### Scenario: Radar input provided with calibration
- **WHEN** a radar tensor and calibration metadata are provided with a vision sample
- **THEN** the system ingests the radar tensor and associates it with the corresponding vision sample

#### Scenario: Radar input provided without calibration
- **WHEN** a radar tensor is provided without calibration metadata
- **THEN** the system rejects the sample with a clear validation error indicating missing calibration

### Requirement: Radar preprocessing and feature extraction
The system SHALL preprocess radar tensors and produce radar feature maps compatible with the vision feature pyramid levels used by Deformable DETR.

#### Scenario: Radar tensor preprocessing
- **WHEN** a valid radar tensor is ingested
- **THEN** the system applies the configured radar preprocessing pipeline and outputs normalized radar features

#### Scenario: Feature pyramid alignment
- **WHEN** radar features are produced
- **THEN** the system outputs radar feature maps aligned in scale and channel dimensions to the vision feature pyramid levels

### Requirement: Fusion of radar and vision features
The system SHALL fuse radar feature maps with vision feature maps prior to attention for detection inference.

#### Scenario: Fusion enabled
- **WHEN** radar fusion is enabled in configuration
- **THEN** the system combines radar and vision feature maps using the configured fusion operator before attention modules execute

#### Scenario: Fusion disabled
- **WHEN** radar fusion is disabled in configuration
- **THEN** the system runs vision-only attention without using radar features

### Requirement: Configuration of radar inputs and fusion
The system SHALL expose configuration fields to enable radar inputs, specify radar preprocessing, and select a fusion operator.

#### Scenario: Radar enabled configuration
- **WHEN** configuration enables radar inputs and sets a fusion operator
- **THEN** the system activates the radar pathway with the specified preprocessing and fusion settings

#### Scenario: Radar disabled configuration
- **WHEN** configuration disables radar inputs
- **THEN** the system ignores any radar tensors and runs with vision-only behavior

### Requirement: Radar-assisted evaluation outputs
The system SHALL report radar-assisted detection metrics alongside existing detection metrics.

#### Scenario: Evaluation with radar enabled
- **WHEN** evaluation is run with radar inputs enabled
- **THEN** the system outputs standard detection metrics and additional radar-assisted metrics in the evaluation report
