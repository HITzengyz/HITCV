## 1. Eval Scale Source and Dataset Flag

- [x] 1.1 Enforce waterscenes `use_size_for_eval=True` in dataset build path
- [x] 1.2 In `engine.evaluate`, select scale source via dataset flag and log source once per eval

## 2. Eval Safety Checks

- [x] 2.1 Add size consistency warning with `image_id`, sample size, target size, orig_size
- [x] 2.2 Add category-domain hard check for predicted labels vs GT category set

## 3. Diagnostics and Sanity Outputs

- [x] 3.1 Save 10 GT+Pred overlay images with scale-source annotation
- [x] 3.2 Print IoU distribution stats (`max_any`, `max_same`: min/median/max)
- [x] 3.3 Print `results.json` bbox sanity stats (xywh validity, out-of-bounds rates)

## 4. Verification

- [ ] 4.1 Verify max IoU max improves beyond 0.3 on same data/checkpoint path
- [ ] 4.2 Verify COCOeval AP > 0 and AR not all zero
