import argparse
import json
import os
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

ONNX_PATH = "models/mlp_asl.onnx"
CLASSES_PATH = "models/mlp_classes.json"
SAMPLES_DIR = "samples"
HAND_MODEL_PATH = "models/hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")
SMOOTH_WINDOW = 10
CONF_THRESH = 0.65


def load_classes(path=CLASSES_PATH):
    with open(path) as f:
        return json.load(f)


def ensure_hand_model(path=HAND_MODEL_PATH):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading hand landmarker model -> {path}")
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
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=det_conf,
        min_hand_presence_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )
    return HandLandmarker.create_from_options(options), mp


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


def draw_confidence_bar(frame, conf, x, y, width=220, height=22):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    fill = int(width * conf)
    green = int(80 + 175 * conf)
    cv2.rectangle(frame, (x, y), (x + fill, y + height), (0, green, 0), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)
    cv2.putText(frame, f"{conf * 100:.1f}%", (x + width + 8, y + height - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description="Real-time ASL landmark MLP demo")
    parser.add_argument("--onnx", default=ONNX_PATH)
    parser.add_argument("--classes", default=CLASSES_PATH)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--det-conf", type=float, default=0.75)
    parser.add_argument("--track-conf", type=float, default=0.6)
    parser.add_argument("--hand-model", default=HAND_MODEL_PATH)
    args = parser.parse_args()

    class_names = load_classes(args.classes)
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    model_path = ensure_hand_model(args.hand_model)
    landmarker, mp = build_landmarker(model_path, args.det_conf, args.track_conf)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return

    history = deque(maxlen=SMOOTH_WINDOW)
    prev_t = time.time()
    fps = 0.0
    saved = 0
    last_ts = 0
    print("Demo running. Press Q to quit, S to save frame.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))
        ts = max(last_ts + 1, int(time.time() * 1000))
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)

        display_label, display_conf = "...", 0.0
        box = None
        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            feat = normalize_landmarks(pts).reshape(1, -1)
            logits = session.run(None, {input_name: feat})[0][0]
            probs = softmax(logits)
            idx = int(probs.argmax())
            history.append((class_names[idx], float(probs[idx])))
            display_label, display_conf = smooth_prediction(history)
            xs = [lm.x * w for lm in lms]
            ys = [lm.y * h for lm in lms]
            box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

        if box is not None:
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if display_label != "..." else (0, 180, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_label, (x1, max(34, y1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
            draw_confidence_bar(frame, display_conf, x1, y2 + 10)

        now = time.time()
        inst = 1.0 / max(1e-6, now - prev_t)
        fps = 0.9 * fps + 0.1 * inst if fps else inst
        prev_t = now
        cv2.putText(frame, f"FPS:{fps:5.1f}", (w - 150, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("ASL Landmark Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("s"), ord("S")):
            fname = os.path.join(
                SAMPLES_DIR, f"lm_{saved:03d}_{display_label}_{display_conf*100:.0f}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
            print(f"Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
