## 1. Diagnostics Tooling

- [x] 1.1 Add script to save 5 GT-only visualizations (transformed RGB + GT boxes)
- [x] 1.2 Add script to save 5 pred+GT visualizations on the same samples
- [x] 1.3 Add script to compute per-image max IoU (same-class and class-agnostic) and print distributions
- [x] 1.4 Log and validate sample/target size conventions (samples H/W, target size/orig_size)

## 2. Alignment Fix (Minimal)

- [x] 2.1 Identify misalignment source (transforms vs target_sizes vs postprocess)
- [x] 2.2 Apply minimal fix in transforms/target sizing to restore alignment

## 3. Verification

- [ ] 3.1 Re-run diagnostics: confirm IoU > 0.3 appears in samples
- [ ] 3.2 Re-run eval: AP/AR not all zeros
