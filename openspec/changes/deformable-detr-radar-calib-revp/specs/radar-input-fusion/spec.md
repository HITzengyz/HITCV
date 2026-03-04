## MODIFIED Requirements

### Requirement: Accept radar input tensors
The system SHALL accept optional radar inputs and calibration metadata alongside existing vision inputs for Deformable DETR training and inference.

#### Scenario: Radar input provided with calibration
- **WHEN** radar data and calibration metadata are provided with a vision sample
- **THEN** the system ingests the radar data, projects it to image space, and associates it with the corresponding vision sample

#### Scenario: Radar input missing
- **WHEN** radar inputs are enabled but no radar data is available for a sample
- **THEN** the system supplies an all-zero radar tensor and sets radar_valid=0 without failing the sample

#### Scenario: Radar input provided without calibration
- **WHEN** radar data is provided without calibration metadata
- **THEN** the system rejects the sample with a clear validation error indicating missing calibration

### Requirement: Radar preprocessing and feature extraction
The system SHALL project radar data to the image plane using calibration metadata and produce REVP multi-channel tensors compatible with the vision feature pyramid levels used by Deformable DETR.

#### Scenario: REVP projection and rasterization
- **WHEN** valid radar data is ingested with calibration
- **THEN** the system projects radar points into the RGB image plane, rasterizes REVP channels (range/doppler/elevation/power) at RGB resolution, and sets radar_valid=1

#### Scenario: Feature pyramid alignment
- **WHEN** radar features are produced
- **THEN** the system outputs radar feature maps aligned in scale and channel dimensions to the vision feature pyramid levels

## ADDED Requirements

### Requirement: REVP collision and normalization policy
The system SHALL apply a deterministic collision and normalization policy when constructing REVP tensors.

#### Scenario: Multi-point collision at a pixel
- **WHEN** multiple radar points map to the same pixel
- **THEN** the system selects the point with maximum power and copies its range/doppler/elevation/power into that pixel

#### Scenario: REVP channel normalization
- **WHEN** REVP tensors are produced
- **THEN** the system applies configured clip+scale normalization per channel before fusion
