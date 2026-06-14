"""
Project 6 - Part 2: YOLO face detection (shared module).

Provides detect_faces(), imported by Parts 3, 5 and 6. The YOLO model is loaded
once at import time so repeated calls are fast.

Weights: yolov8n-face.pt from akanametov/yolo-face (release 1.0.0), <7 MB.
    https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt

CLI:
    python detect.py <image> [--conf 0.4] [--out annotated.jpg]
"""

import os
import argparse
import cv2
from ultralytics import YOLO

# Resolve the weights path relative to this file so imports work from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(_HERE, "models", "yolov8n-face.pt")

# Load the detector once (module-level singleton).
face_detector = YOLO(WEIGHTS)


def detect_faces(bgr_image, conf=0.4):
    """Return a list of (x1, y1, x2, y2, conf) face boxes, ints + float conf."""
    results = face_detector.predict(bgr_image, conf=conf, verbose=False)
    boxes = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            boxes.append((int(x1), int(y1), int(x2), int(y2), float(b.conf[0])))
    return boxes


def draw_boxes(bgr_image, boxes):
    """Return a copy of the image with green boxes + confidence labels drawn."""
    out = bgr_image.copy()
    for (x1, y1, x2, y2, c) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{c:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to an image")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--out", default=None, help="output path (default: <name>_detected.jpg)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    boxes = detect_faces(img, conf=args.conf)
    print(f"{len(boxes)} face(s) detected in {args.image}:")
    for i, (x1, y1, x2, y2, c) in enumerate(boxes):
        print(f"  [{i}] box=({x1},{y1},{x2},{y2}) conf={c:.3f}")

    out_path = args.out or os.path.splitext(args.image)[0] + "_detected.jpg"
    cv2.imwrite(out_path, draw_boxes(img, boxes))
    print("annotated image saved to:", out_path)
