## MODIFIED Requirements

### Requirement: Accept optional TIR input
The system SHALL accept an optional thermal infrared (TIR) image path per COCO image entry and SHALL normalize labels to a 0-based contiguous class index space during dataset preparation.

#### Scenario: TIR path provided
- **WHEN** a COCO `images` entry includes `tir_file`
- **THEN** the system loads the grayscale TIR image and associates it with the RGB image

#### Scenario: TIR path missing
- **WHEN** a COCO `images` entry omits `tir_file`
- **THEN** the system proceeds with RGB-only input and marks TIR as invalid

#### Scenario: Labels are 1-based
- **WHEN** COCO `annotations` contain `category_id` values that start at 1
- **THEN** the system normalizes them to 0-based contiguous IDs and writes a new annotation JSON without modifying the original

#### Scenario: Labels are non-contiguous
- **WHEN** COCO `annotations` contain gaps in `category_id` values
- **THEN** the system remaps them to contiguous IDs in `[0, C-1]` and writes a new annotation JSON without modifying the original

### Requirement: Backbone input compatibility
The system SHALL validate the input tensor shape `[5, H, W]` prior to the backbone and SHALL allow the class count used by model heads, matcher, and losses to be explicitly configured.

#### Scenario: Forward pass input validation
- **WHEN** a sample is forwarded to the backbone
- **THEN** the system asserts the input tensor has 5 channels and rejects other shapes

#### Scenario: Class count override
- **WHEN** a user specifies `--num_classes <C>`
- **THEN** the model classification heads, matcher, and loss functions use `<C>` consistently
