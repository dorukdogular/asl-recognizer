---
title: ASL Hand Gesture Recognizer
emoji: 🤟
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
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
scale-normalized hand landmarks (`x,y,z` × 21). The landmark MLP reaches
99%+ (landmark MLP, val set) and is used here for robust live webcam performance.

## Tech stack
- **Gradio** — streaming webcam UI
- **MediaPipe Hands** — 21-point hand landmark detection
- **ONNX Runtime** — CPU inference
- **OpenCV / NumPy** — frame annotation and preprocessing

## Dataset & Credits

**Dataset:** ASL-HG — American Sign Language Hand Gesture Image Dataset  
Pranto et al. (2026), Data in Brief  
https://www.sciencedirect.com/science/article/pii/S2352340926000454  
DOI: https://doi.org/10.17632/j4y5w2c8w9.1

**Model & App:** Doruk Doğular (nocontextdoruk)  
Landmark-based MLP trained on ASL-HG processed split.
