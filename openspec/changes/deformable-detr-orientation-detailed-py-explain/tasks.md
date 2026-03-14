## 1. Scope and Inventory

- [x] 1.1 Extract the exact section-2 core/support Python file list from `detr/docs/codebase_orientation.md`
- [x] 1.2 Verify each listed file exists and freeze the coverage baseline for this change

## 2. Per-file Deep Explanations

- [x] 2.1 Add detailed subsections for all section-2 core Python files (role, key symbols, call graph)
- [x] 2.2 Add detailed subsections for all section-2 runtime support Python files (role, key symbols, call graph)
- [x] 2.3 Ensure each subsection is beginner-oriented and references concrete source file paths

## 3. Shape Annotation Pass

- [x] 3.1 Add shape propagation notes for key data paths where static code evidence exists
- [x] 3.2 Mark unknown/non-inferable dimensions explicitly as unknown from static inspection
- [x] 3.3 Remove any unsupported concrete shape claims that cannot be justified from code

## 4. Consistency and Readability Validation

- [x] 4.1 Validate the section-2 summary table and per-file subsections are one-to-one aligned
- [x] 4.2 Validate all referenced file paths and symbol names against current code tree
- [x] 4.3 Review document flow for newcomer readability (quick map -> deep detail -> shape caveats)
