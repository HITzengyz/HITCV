# Spec: Detection Input Modalities

## Purpose
Define requirements for ingesting multi-modal inputs (RGB + optional TIR) for Deformable DETR training and inference.

## Requirements

### Requirement: Accept optional TIR input
The system SHALL accept an optional thermal infrared (TIR) image path per COCO image entry.

#### Scenario: TIR path provided
- **WHEN** a COCO `images` entry includes `tir_file`
- **THEN** the system loads the grayscale TIR image and associates it with the RGB image

#### Scenario: TIR path missing
- **WHEN** a COCO `images` entry omits `tir_file`
- **THEN** the system proceeds with RGB-only input and marks TIR as invalid

### Requirement: Produce 5-channel input tensor
The system SHALL construct a 5-channel input tensor `[R, G, B, TIR, tir_valid]` with shape `[5, H, W]`.

#### Scenario: TIR available
- **WHEN** a valid TIR image is loaded
- **THEN** the system normalizes the TIR channel and sets `tir_valid` to 1 across `[1, H, W]`

#### Scenario: TIR unavailable or unreadable
- **WHEN** the TIR image is missing or unreadable
- **THEN** the system sets the TIR channel to zeros and `tir_valid` to 0 across `[1, H, W]`

### Requirement: Apply spatial transforms consistently
The system SHALL apply spatial transforms to RGB and TIR channels consistently.

#### Scenario: Spatial transforms configured
- **WHEN** resize/crop/flip/pad are applied to the sample
- **THEN** the system applies identical transforms to RGB and TIR channels, preserving alignment

### Requirement: Support TIR modality dropout
The system SHALL support optional TIR-only modality dropout during training.

#### Scenario: Modality dropout enabled
- **WHEN** `--tir_dropout` is set to a value in `[0, 1]`
- **THEN** the system randomly zeroes the TIR channel and sets `tir_valid=0` with that probability

### Requirement: Backbone input compatibility
The system SHALL validate the input tensor shape `[5, H, W]` prior to the backbone.

#### Scenario: Forward pass input validation
- **WHEN** a sample is forwarded to the backbone
- **THEN** the system asserts the input tensor has 5 channels and rejects other shapes
