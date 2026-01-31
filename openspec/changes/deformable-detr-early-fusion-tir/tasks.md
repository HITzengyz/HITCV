# Tasks

- [x] Update dataset loader to read optional COCO `tir_file` and return TIR tensor + tir_valid
- [x] Apply spatial transforms to TIR (resize/crop/flip/pad) in transforms pipeline
- [x] Concatenate RGB+TIR+tir_valid into 5-channel tensor and update normalization to 5 channels
- [x] Add Modality Dropout (TIR only) with configurable `--tir_dropout` default 0.3
- [x] Update backbone conv1 in_channels from 3 to 5
- [x] Add forward/input shape assertion for [5, H, W]
- [x] Add unit/script checks for shape/range/tir_valid logic
- [x] Add missing-TIR inference test (no crash)
- [x] Run 1-epoch smoke test and record log/visual proof of 5-channel input
