---
title: ASL Hand Gesture Recognizer
emoji: 🤟
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# ASL Hand Gesture Recognizer

## What it does
Real-time American Sign Language recognition from your webcam. Each frame runs
MediaPipe hand tracking, extracts 21 normalized hand landmarks, and classifies the
gesture with a lightweight MLP exported to ONNX. Predictions are smoothed over the
last 10 frames so the label stays stable while you hold a sign.

## Classes
36 total: letters **A–Z** and digits **0–9**.

## How to use
1. Allow camera access.
2. Show your hand to the webcam.
3. Hold the gesture steady — the predicted letter/digit appears top-left with a
   confidence bar. Landmark dots are drawn on your hand. Low-confidence frames show
   `...` until the prediction settles.

## Model
MLP (63 → 256 → 128 → 36) trained on the ASL-HG dataset. Input is wrist-centered,
scale-normalized hand landmarks (`x,y,z` × 21). Held-out validation accuracy is
**100%** on 6,193 landmark samples — the landmark representation makes the 36
classes near-linearly separable, and it avoids the live-webcam distribution shift
that hurt the earlier image-CNN baseline.

## Results & Analysis

Evaluated on the stratified 20% validation split (6,193 samples).

| Metric | Value |
|---|---|
| Validation accuracy | 100.00% |
| Macro per-class accuracy | 100.00% |
| Classes | 36 (A–Z, 0–9) |
| Landmark samples | 30,962 (hand detected) |

![Confusion matrix](analysis/confusion_matrix.png)
![Per-class accuracy](analysis/per_class_accuracy.png)
![Samples per class](analysis/class_distribution.png)

The confusion matrix is fully diagonal. `class_distribution.png` shows how many
images yielded a detected hand per class (closed-fist signs like `O`, `C`, `A`, `T`
retain fewer samples because MediaPipe detects them less often).

## Tech stack
- **Gradio** — streaming webcam UI
- **MediaPipe Hands** — 21-point hand landmark detection
- **ONNX Runtime** — CPU inference
- **OpenCV / NumPy** — frame annotation and preprocessing

## Dataset & Credits

**Dataset:** ASL-HG — American Sign Language Hand Gesture Image Dataset  
Pranto et al. (2026), Data in Brief  
Article: https://www.sciencedirect.com/science/article/pii/S2352340926000454  
Mendeley Data: https://data.mendeley.com/datasets/j4y5w2c8w9/1  
DOI: https://doi.org/10.17632/j4y5w2c8w9.1

**Model & App:** Doruk Doğular (nocontextdoruk)  
Landmark-based MLP trained on ASL-HG processed split.
Model repo: https://huggingface.co/nocontextdoruk/asl-landmark-mlp

## License

**CC BY-NC 4.0** — free for research, education, and personal/open projects **with
attribution**; **no commercial or enterprise resale**. See [LICENSE](LICENSE).
If you use it, please cite via [CITATION.cff](CITATION.cff). The ASL-HG dataset is
owned by its original authors (cite separately).
