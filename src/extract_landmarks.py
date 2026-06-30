import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_DATA_ROOT = "data/asl_processed"
OUTPUT_PATH = "data/landmarks.npz"
HAND_MODEL_PATH = "models/hand_landmarker.task"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_class_dirs(root):
    root = Path(root)
    return sorted([d for d in root.iterdir()
                   if d.is_dir() and any(f.suffix.lower() in IMG_EXTS for f in d.iterdir())],
                  key=lambda d: d.name)


def discover_samples(data_root):
    root = Path(data_root)
    splits = [root / "train", root / "test"]
    if not all(s.is_dir() for s in splits):
        splits = [root]
    class_to_files = {}
    for split in splits:
        for cdir in list_class_dirs(split):
            files = [f for f in cdir.iterdir() if f.suffix.lower() in IMG_EXTS]
            class_to_files.setdefault(cdir.name, []).extend(files)
    class_names = sorted(class_to_files)
    samples = []
    for label, name in enumerate(class_names):
        for f in sorted(class_to_files[name]):
            samples.append((str(f), label, name))
    return class_names, samples


def normalize_landmarks(pts):
    wrist = pts[0]
    centered = pts - wrist
    scale = np.linalg.norm(centered[9])
    if scale < 1e-6:
        scale = 1.0
    return (centered / scale).flatten().astype(np.float32)


def build_landmarker(model_path, det_conf):
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
    )
    return HandLandmarker.create_from_options(options)


def main():
    parser = argparse.ArgumentParser(description="Extract normalized hand landmarks")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--hand-model", default=HAND_MODEL_PATH)
    parser.add_argument("--det-conf", type=float, default=0.5)
    args = parser.parse_args()

    import mediapipe as mp

    class_names, samples = discover_samples(args.data_root)
    print(f"Classes: {len(class_names)}  Images: {len(samples)}")
    landmarker = build_landmarker(args.hand_model, args.det_conf)

    X, y = [], []
    skipped = {name: 0 for name in class_names}
    kept = {name: 0 for name in class_names}

    for path, label, name in tqdm(samples, desc="extract"):
        img = cv2.imread(path)
        if img is None:
            skipped[name] += 1
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))
        result = landmarker.detect(mp_image)
        if not result.hand_landmarks:
            skipped[name] += 1
            continue
        lms = result.hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
        X.append(normalize_landmarks(pts))
        y.append(label)
        kept[name] += 1

    landmarker.close()
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, X=X, y=y, class_names=np.array(class_names))

    total_skip = sum(skipped.values())
    print("=" * 44)
    print(f"{'class':<8}{'kept':>8}{'skipped':>10}")
    for name in class_names:
        print(f"{name:<8}{kept[name]:>8}{skipped[name]:>10}")
    print("-" * 44)
    print(f"Saved {args.output}: X={X.shape} y={y.shape}")
    print(f"Total kept={len(y)}  total skipped={total_skip}")
    print("=" * 44)


if __name__ == "__main__":
    main()
