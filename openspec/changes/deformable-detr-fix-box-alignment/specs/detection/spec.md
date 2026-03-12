## ADDED Requirements

### Requirement: Visualize transformed GT boxes
The system SHALL save RGB images with transformed GT boxes overlaid for a fixed set of training samples.

#### Scenario: GT overlay export
- **WHEN** a diagnostic run is executed
- **THEN** the system writes at least 5 images with RGB + GT boxes derived from `target["boxes"]`

### Requirement: Visualize predictions alongside GT
The system SHALL save images with both predictions and GT boxes overlaid for the same samples.

#### Scenario: Pred+GT overlay export
- **WHEN** a diagnostic run is executed
- **THEN** the system writes at least 5 images with prediction boxes and GT boxes on the same image

### Requirement: IoU distribution analysis
The system SHALL compute max IoU per image for both class-agnostic and same-class matching.

#### Scenario: IoU stats output
- **WHEN** a diagnostic run is executed
- **THEN** the system prints distributions of max IoU (min/median/max) for both modes

### Requirement: Size convention logging
The system SHALL log sample and target size conventions for validation.

#### Scenario: Size logs
- **WHEN** a diagnostic run is executed
- **THEN** the system logs `samples(H,W)`, `target["size"]`, and `target["orig_size"]` and confirms order `[h,w]`
