# Active Context

## Current Focus
V2 landmark-MLP COMPLETE. Pivoted off EfficientNet (failed live webcam, distro
shift). Pipeline now: image/frame → MediaPipe Hands → 21 (x,y,z) landmarks →
normalize (wrist-centered, scale ‖lm9−lm0‖) → MLP. val_acc=1.0000.
Files: `src/{extract_landmarks,train_mlp,demo_landmark}.py`. Old src deleted.
Run live local: `.venv/bin/python src/demo_landmark.py` (needs camera permission).

## Deploy — HuggingFace Spaces (Gradio)
`app.py` at root + flat assets (`mlp_asl.onnx`, `mlp_classes.json`,
`hand_landmarker.task` copied to root). Root `requirements.txt` = Spaces deps
(gradio, onnxruntime, mediapipe, opencv-python-headless, numpy). Root `README.md`
= Spaces frontmatter. Push root to a Spaces repo to deploy. Note: root
`requirements.txt`/`README.md` now serve Spaces (training-time docs superseded).

## Dataset Discovery Result
- Detected **pre-split** layout at `data/asl_processed/{train,test}`.
- 36 classes (`0-9`, `A-Z`); train 800/class (28,800), test 200/class (7,200).
- Distribution balanced → use as-is, no programmatic split, no file moves.

## Environment Notes
- System Python 3.14 (no torch/mediapipe wheels yet).
- Runtime use dedicated `python3.12` venv at `.venv/` (see README).
- Apple Silicon: training/eval auto-select `mps` if available, else `cpu`.

## Decisions
- `class_to_idx` from sorted class names; persisted to `class_map.json`.
- Images resized 224 (val: Resize(256)+CenterCrop(224)), ImageNet normalization.
- mediapipe 0.10.35 = Tasks-only (no `mp.solutions`). Demo use `HandLandmarker`
  Tasks API, VIDEO mode; `models/hand_landmarker.task` auto-downloads.

## Demo (rewritten for speed + stability)
- Inference via `onnxruntime` on `models/asl_classifier.onnx` (no torch in demo).
- Crop = landmark bbox expanded 20% each side (not flat 30px), resize 224
  `INTER_LANCZOS4`, ImageNet Normalize.
- Smoothing: deque last 10 `(label, conf)`, confidence-weighted vote; show label
  only if smoothed top-1 conf > 0.65 else `...`; conf shown as %.
- HandLandmarker det_conf=0.75, track_conf=0.6 (full bundle = model_complexity 1).
- Headless smoke test passed (preproc, ONNX infer, expand_box, smoother, landmarker).

## Next Steps
- Grant camera permission (System Settings → Privacy → Camera) then run demo.
- Optional: resume training to push past 93.44%.