# Project Brief — ASL Hand Gesture Recognition

## Goal
Production-quality American Sign Language (ASL) static hand-gesture classifier. 36 classes (digits `0-9` + letters `A-Z`). Pipeline: dataset discovery, training, evaluation, real-time webcam demo, ONNX export.

## Dataset
- Location: `data/asl_processed/`
- Structure: **pre-split** `train/` and `test/`, one folder per class.
- Counts: 28,800 train (800/class) + 7,200 test (200/class), perfectly balanced.
- Images: RGB `.jpg`, used as-is (no file moves). Pre-split ratio ≈ 80/20.

## Model
- Backbone: EfficientNet-B0 pretrained on ImageNet.
- Head: `Dropout(0.3) -> Linear(1280, 36)`.
- Two-phase training:
  - Phase 1 (epochs 1-8): backbone frozen, head only, lr=1e-3.
  - Phase 2 (epochs 9-25): full unfreeze, lr=1e-4, layer-wise decay (backbone lr*0.1, head lr).
- Optimizer AdamW, CosineAnnealingLR per phase, early stopping patience=5 on val_loss.

## Quality Bars
- Min test accuracy: **92%**.
- If unmet after full training: auto Phase 3 = mixup augmentation + 10 extra epochs.
- Demo must sustain **≥15 FPS on CPU**.

## Deliverables
`src/{dataset,model,train,evaluate,demo,export}.py`, `requirements.txt`, `README.md`, artifacts in `checkpoints/ logs/ results/ models/ samples/`, and `class_map.json`.