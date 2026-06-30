# Progress

## Done
- [x] Dataset discovery: pre-split, balanced, 36 classes, 28,800/7,200.
- [x] Project structure made (`memory-bank src checkpoints logs results models samples`).
- [x] memory-bank init.

- [x] Source files (`dataset, model, train, evaluate, demo, export`).
- [x] Env setup (python3.12 venv at `.venv/` + deps).
- [x] Sanity check (1-batch forward pass): loss 3.62 ≈ ln(36).
- [x] Phase 1 + Phase 2 train (stopped early e12, bar already met).
- [x] Eval: **93.44%** test accuracy (target 92% MET).
- [x] ONNX export + verify (single file, opset 17, argmax parity).

## V2 — Landmark MLP (CURRENT approach)
Reason: EfficientNet+crop failed on live webcam (distribution shift). Switched
to MediaPipe-landmark features → MLP. Old src files deleted.

- [x] `extract_landmarks.py`: 36,000 imgs → MediaPipe Hands (IMAGE, det 0.5),
  21 (x,y,z) landmarks, normalize (subtract wrist lm0, divide by ‖lm9−lm0‖).
  Output `data/landmarks.npz` X=(30962,63) f32, y, class_names. Skipped 5038
  (no hand); lowest retention O=296, C=379, A=427, T=426; full classes B/2/3/4/etc.
- [x] `train_mlp.py`: MLP 63→256→BN→ReLU→Drop0.3→128→BN→ReLU→Drop0.2→36.
  AdamW lr1e-3, CosineAnnealing, 100 ep, batch256, stratified 80/20.
  **val_acc=1.0000** (target 0.95 MET). Saved `models/mlp_best.pth`,
  `models/mlp_asl.onnx` (opset17), `models/mlp_classes.json`.
- [x] `demo_landmark.py`: webcam → HandLandmarker VIDEO (det0.75/track0.6) →
  normalize → MLP ONNX → softmax → deque10 weighted vote, thresh0.65, "...".
  Overlay class/conf%/FPS. ONNX↔torch argmax parity 1.0.
- [x] Deleted old src: demo, export, train, evaluate, dataset, model.

## V3 — HuggingFace Spaces deploy (Gradio)
- [x] `app.py`: `gr.Interface` webcam stream (`sources=["webcam"]`, streaming,
  live=True). Per-frame: MediaPipe HandLandmarker IMAGE (det0.75) → normalize →
  `mlp_asl.onnx` → softmax → module-global deque10 weighted smoother (thresh0.65).
  Annotates: big label top-left, conf% bar, landmark dots; "No hand detected" else.
  gradio version-robust (`flagging_mode` v5+ else `allow_flagging`).
- [x] Flat root layout for Spaces: copied `mlp_asl.onnx`, `mlp_classes.json`,
  `hand_landmarker.task` to project root; `app.py` reads root paths; `.task`
  auto-downloads if missing.
- [x] `requirements.txt` (Spaces): gradio>=4.0, onnxruntime, mediapipe,
  opencv-python-headless, numpy. `README.md` = Spaces frontmatter (sdk gradio 4.44.0).
- [x] Smoke test passed (import, no-hand frame, onnx infer, live=True).

## V4 — Live deploy (HF Space + model repo)
- [x] Space **RUNNING**: https://huggingface.co/spaces/nocontextdoruk/asl-recognizer (HTTP 200).
- [x] Model repo: https://huggingface.co/nocontextdoruk/asl-landmark-mlp
  (mlp_asl.onnx + mlp_classes.json + model card, Mendeley cite).
- [x] Mendeley Data cite explicit in Space README + model card
  (https://data.mendeley.com/datasets/j4y5w2c8w9/1, DOI 10.17632/j4y5w2c8w9.1).
- Build fixes (sequential): audioop removed in py3.13 → `python_version: "3.12"`;
  gradio 4.44 imports `HfFolder` (gone in huggingface_hub≥1.0) → `gradio==5.49.1`;
  mediapipe native lib needs `libGLESv2.so.2` → `packages.txt` (libgl1, libglib2.0-0,
  libgles2, libegl1).
- Published via huggingface_hub API (cached token).
- [x] GitHub: https://github.com/dorukdogular/asl-recognizer (code only, 844 KB;
  dataset + `*.task` gitignored — hand model auto-downloads).
- [x] Live webcam bug fixed: empty predictions were the fragile `gr.Interface(live=True)`
  streaming path → rewrote `app.py` to `gr.Blocks` + `webcam.stream(...)` +
  `demo.queue()`, defensive frame handling (RGBA/gray/None), stream_every=0.15.
  Inference logic was correct (real imgs → right labels conf~1.0). Space re-verified RUNNING.
- [x] License **CC BY-NC 4.0** (non-commercial + attribution) in LICENSE, CITATION.cff,
  and all READMEs (Space/model/GitHub). HF frontmatter `license: cc-by-nc-4.0`.
- [x] Analysis: `analysis/{confusion_matrix,per_class_accuracy,class_distribution}.png`
  + metrics.json. Val acc **100%** (6193 samples, fully diagonal — landmarks separable).
  Embedded in Space + model READMEs.
- [x] Cleanup: removed ~110M obsolete EfficientNet artifacts (checkpoints/,
  asl_classifier.onnx), old logs/results, caches, class_map.json.

## V1 — EfficientNet (DEPRECATED, files removed)
- Two-phase EfficientNet-B0, test acc 93.44%. Failed live (train/webcam shift).
- Artifacts `checkpoints/`, `models/asl_classifier.onnx` still on disk (unused).