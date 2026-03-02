## ADDED Requirements

### Requirement: Provide runtime-path architecture map
The project SHALL provide a documentation artifact that explains the runtime path for Deformable-DETR from training entry to model outputs.

#### Scenario: Entry path for training/evaluation
- **WHEN** a reader opens the orientation document
- **THEN** they can trace `main.py -> models/build -> model forward/loss -> engine evaluate` with concrete file references

### Requirement: Distinguish core, support, and generated code areas
The project SHALL classify directories/files into core model mechanism, runtime support, and non-core generated artifacts.

#### Scenario: Core vs non-core boundary
- **WHEN** a reader needs to decide where to start or what to edit
- **THEN** the document identifies which paths are model-critical and which are outputs/log artifacts

### Requirement: Provide novice-first reading order
The project SHALL include a beginner reading sequence to reduce onboarding friction.

#### Scenario: First-time model reader
- **WHEN** a reader is unfamiliar with Deformable-DETR
- **THEN** they receive an ordered reading path from high-level entrypoints to detailed modules

### Requirement: Include safe editing guidance
The project SHALL include practical editing boundaries for contributors.

#### Scenario: Contributor plans a change
- **WHEN** a contributor chooses files to modify
- **THEN** the guide clearly marks "edit-first" locations and "avoid-editing-first" generated/artifact locations
