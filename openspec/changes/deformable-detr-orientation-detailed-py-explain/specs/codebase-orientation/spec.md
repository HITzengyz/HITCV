## MODIFIED Requirements

### Requirement: Distinguish core, support, and generated code areas
The project SHALL classify directories/files into core model mechanism, runtime support, and non-core generated artifacts, and SHALL pair each core/support Python file listed in the orientation table with a dedicated explanatory subsection.

#### Scenario: Core vs non-core boundary
- **WHEN** a reader needs to decide where to start or what to edit
- **THEN** the document identifies which paths are model-critical and which are outputs/log artifacts

#### Scenario: Per-file subsection coverage
- **WHEN** a Python file is listed under section-2 core or runtime support categories
- **THEN** the orientation document includes a dedicated subsection for that file

## ADDED Requirements

### Requirement: Provide detailed per-file explanations for section-2 core/support Python files
The project SHALL provide detailed, beginner-oriented explanations for each section-2 core/support Python file, including role, key symbols, and call relationships.

#### Scenario: Reader opens a file subsection
- **WHEN** a reader opens the subsection for a listed Python file
- **THEN** they can see the file's pipeline role, key classes/functions, and upstream/downstream call context

#### Scenario: Cross-file navigation
- **WHEN** a reader follows the runtime path
- **THEN** the document provides enough file-level context to continue from `main.py` through model, dataset, and engine components without external assumptions

### Requirement: Shape annotations MUST be evidence-based
The project SHALL annotate tensor/matrix shapes only when statically inferable from code contracts, assertions, or explicit construction logic, and SHALL avoid fabricated concrete dimensions.

#### Scenario: Shape inferable from code
- **WHEN** a shape can be derived from explicit code constraints or symbolic contracts
- **THEN** the document reports the shape using symbolic or concrete notation and cites the governing source context

#### Scenario: Shape not inferable from static inspection
- **WHEN** a dimension cannot be justified from current code/data contracts
- **THEN** the document explicitly marks it as unknown from static inspection and does not guess
