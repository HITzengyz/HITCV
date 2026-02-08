## 1. Dataset Label Normalization

- [x] 1.1 Add a utility/script to generate 0-based contiguous COCO JSONs without overwriting originals
- [x] 1.2 Validate generated JSONs: category_id range, contiguous mapping, counts preserved

## 2. Training Class Count Override

- [x] 2.1 Add `--num_classes` CLI arg and thread it into model build
- [x] 2.2 Ensure matcher/loss/class head all use the same `num_classes` value

## 3. Minimal Acceptance Checks

- [ ] 3.1 Run 1–5 image overfit sanity (loss decreases, predictions non-empty)
- [x] 3.2 Document the recommended training invocation for waterscenes (num_classes=7 + normalized JSON)
