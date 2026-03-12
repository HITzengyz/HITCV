## ADDED Requirements

### Requirement: Classification head includes no-object class
The system SHALL output `num_classes + 1` classification logits per query, where the final index represents the no-object class.

#### Scenario: Forward pass classification logits
- **WHEN** a batch is forwarded through the model
- **THEN** `pred_logits` has shape `[B, num_queries, num_classes + 1]` and no-object index is `num_classes`

### Requirement: Criterion uses no-object weighting
The system SHALL apply an `empty_weight` of length `num_classes + 1` with `empty_weight[-1] = eos_coef` when computing classification loss.

#### Scenario: Classification loss weighting
- **WHEN** classification loss is computed
- **THEN** the no-object class is weighted by `eos_coef` and all other classes use weight 1

### Requirement: Post-process excludes no-object
The system SHALL exclude the no-object class from detection results.

#### Scenario: Exporting detection results
- **WHEN** results are exported for evaluation
- **THEN** only probabilities over foreground classes `[0, num_classes - 1]` are used

### Requirement: Eval sanity logging
The system SHALL print eval-time sanity diagnostics for no-object handling.

#### Scenario: Eval debug output
- **WHEN** an evaluation run starts
- **THEN** the system logs `pred_logits` shape, `noobj_idx`, and `p_noobj`
