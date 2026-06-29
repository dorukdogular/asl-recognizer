import json
import os
import urllib.request
from collections import deque

import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort

ONNX_PATH = "mlp_asl.onnx"
CLASSES_PATH = "mlp_classes.json"
HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")
SMOOTH_WINDOW = 10
CONF_THRESH = 0.65


def ensure_hand_model(path=HAND_MODEL_PATH):
    if not os.path.exists(path):
        urllib.request.urlretrieve(HAND_MODEL_URL, path)
    return path


def build_landmarker(model_path, det_conf=0.75, track_conf=0.6):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker, HandLandmarkerOptions, RunningMode,
    )
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=det_conf,
        min_hand_presence_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )
    return HandLandmarker.create_from_options(options), mp


with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

SESSION = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = SESSION.get_inputs()[0].name
LANDMARKER, MP = build_landmarker(ensure_hand_model())
HISTORY = deque(maxlen=SMOOTH_WINDOW)


def normalize_landmarks(pts):
    wrist = pts[0]
    centered = pts - wrist
    scale = np.linalg.norm(centered[9])
    if scale < 1e-6:
        scale = 1.0
    return (centered / scale).flatten().astype(np.float32)


def softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def smooth_prediction(history):
    scores = {}
    for label, conf in history:
        scores[label] = scores.get(label, 0.0) + conf
    total = sum(scores.values())
    if total <= 0:
        return "...", 0.0
    winner = max(scores, key=scores.get)
    smooth_conf = scores[winner] / total
    if smooth_conf < CONF_THRESH:
        return "...", smooth_conf
    return winner, smooth_conf


def draw_landmarks(frame, lms, w, h, color=(0, 255, 0)):
    for lm in lms:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, color, -1)


def draw_confidence_bar(frame, conf, x, y, width=240, height=24):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    fill = int(width * conf)
    green = int(80 + 175 * conf)
    cv2.rectangle(frame, (x, y), (x + fill, y + height), (0, green, 0), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)
    cv2.putText(frame, f"{conf * 100:.1f}%", (x + width + 10, y + height - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def predict(frame):
    if frame is None:
        return None
    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]
    mp_image = MP.Image(image_format=MP.ImageFormat.SRGB, data=frame)
    result = LANDMARKER.detect(mp_image)

    if not result.hand_landmarks:
        HISTORY.clear()
        cv2.putText(frame, "No hand detected", (20, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 80, 80), 3)
        return frame

    lms = result.hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    logits = SESSION.run(None, {INPUT_NAME: normalize_landmarks(pts).reshape(1, -1)})[0][0]
    probs = softmax(logits)
    idx = int(probs.argmax())
    HISTORY.append((CLASS_NAMES[idx], float(probs[idx])))
    label, conf = smooth_prediction(HISTORY)

    draw_landmarks(frame, lms, w, h)
    color = (0, 255, 0) if label != "..." else (255, 180, 0)
    cv2.putText(frame, label, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)
    draw_confidence_bar(frame, conf, 20, 90)
    return frame


_flag_kw = ({"flagging_mode": "never"}
            if int(gr.__version__.split(".")[0]) >= 5
            else {"allow_flagging": "never"})

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(sources=["webcam"], streaming=True, label="Webcam"),
    outputs=gr.Image(label="Prediction"),
    live=True,
    title="ASL Hand Gesture Recognizer",
    description="ASL hand gesture recognizer (A–Z, 0–9). Dataset: ASL-HG by Pranto et al. (2026) — https://www.sciencedirect.com/science/article/pii/S2352340926000454 | Model by Doruk Doğular",
    **_flag_kw,
)


if __name__ == "__main__":
    demo.launch()
