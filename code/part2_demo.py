"""
Project 6 - Part 2 deliverables.

1. Run detect_faces on three database images (incl. a low-light one) -> figures.
2. Build a challenging group collage from synthetic faces (varied scale, a
   low-light face, an occluded face, a rotated/profile-like face), detect, and
   compare detected count vs known ground truth.

Outputs land in ../report/.
"""

import os
import glob
import cv2
import numpy as np
from detect import detect_faces, draw_boxes

REPORT = os.path.join(os.path.dirname(__file__), "..", "report")
DB = os.path.join(os.path.dirname(__file__), "..", "database")
CONF = 0.4


def demo_database_images():
    """Detect faces on three database images and save annotated outputs."""
    ids = sorted(glob.glob(os.path.join(DB, "*", "")))
    # pick three: an original, a zoom view, and a LOW-LIGHT view (stress test)
    picks = [
        (ids[0], "00.jpg", "original"),
        (ids[1], "02.jpg", "zoom-crop"),
        (ids[2], "03.jpg", "low-light"),
    ]
    print("=== Part 2.1: detection on three database images ===")
    for i, (d, fname, tag) in enumerate(picks):
        img = cv2.imread(os.path.join(d, fname))
        boxes = detect_faces(img, conf=CONF)
        out = draw_boxes(img, boxes)
        name = os.path.basename(d.rstrip("/"))
        path = os.path.join(REPORT, f"part2_db{i+1}_{tag}.png")
        cv2.imwrite(path, out)
        conf_str = ", ".join(f"{b[4]:.2f}" for b in boxes) or "-"
        print(f"  {name}/{fname} ({tag}): {len(boxes)} face(s) [conf: {conf_str}] -> {path}")


def _place(canvas, face, x, y, w, h):
    """Resize face to (w,h) and paste at (x,y), clipping to canvas bounds."""
    H, W = canvas.shape[:2]
    face = cv2.resize(face, (w, h))
    x2, y2 = min(x + w, W), min(y + h, H)
    x1, y1 = max(x, 0), max(y, 0)
    canvas[y1:y2, x1:x2] = face[y1 - y:y2 - y, x1 - x:x2 - x]


def build_group_collage():
    """Compose a deliberately challenging group photo with known ground truth.

    Eight faces spanning easy -> hard so we can honestly observe which
    conditions YOLO handles and which it misses (scale, lighting, rotation,
    occlusion, glare).
    """
    ids = sorted(glob.glob(os.path.join(DB, "*", "")))
    canvas = np.full((640, 1000, 3), 45, np.uint8)

    def src(i, fname="00.jpg"):
        return cv2.imread(os.path.join(ids[i], fname))

    # --- 8 faces, increasing difficulty (ground truth = 8) ---
    _place(canvas, src(3),   20,  40, 220, 220)             # 1: large, clear (easy)
    _place(canvas, src(4),  270,  60, 140, 140)             # 2: medium, clear

    small = cv2.resize(src(5), (55, 55))                    # 3: tiny face (~55px)
    _place(canvas, small, 440, 70, 55, 55)

    dark = np.clip(src(6).astype(np.float32) * 0.18, 0, 255).astype(np.uint8)
    _place(canvas, dark, 540, 40, 180, 180)                 # 4: heavy low-light (x0.18)

    prof = cv2.warpAffine(                                  # 5: rotated 80 deg (sideways/profile-like)
        cv2.resize(src(7), (180, 180)),
        cv2.getRotationMatrix2D((90, 90), 80, 1.0), (180, 180))
    _place(canvas, prof, 760, 60, 180, 180)

    _place(canvas, src(8), 120, 360, 200, 200)              # 6: base face...
    occ = src(9)                                            # 7: ...heavily occluding #6 (~60% overlap)
    _place(canvas, occ, 250, 400, 150, 150)

    glare = src(10).copy().astype(np.float32)               # 8: blown-out glare on right half
    glare[:, glare.shape[1] // 2:] += 170
    glare = np.clip(glare, 0, 255).astype(np.uint8)
    _place(canvas, glare, 480, 400, 190, 190)

    # --- 2 expected-hard cases (likely missed) ---
    sideways = cv2.warpAffine(                              # 9: rotated 90 deg (fully sideways)
        cv2.resize(src(11), (170, 170)),
        cv2.getRotationMatrix2D((85, 85), 90, 1.0), (170, 170))
    _place(canvas, sideways, 700, 380, 170, 170)

    heavy_occ = cv2.resize(src(12), (170, 170))             # 10: ~85% occluded (only eye strip shows)
    heavy_occ[:55, :] = 45
    heavy_occ[95:, :] = 45
    _place(canvas, heavy_occ, 360, 200, 170, 170)

    cv2.imwrite(os.path.join(REPORT, "part2_group_raw.png"), canvas)

    boxes = detect_faces(canvas, conf=CONF)
    out = draw_boxes(canvas, boxes)
    cv2.imwrite(os.path.join(REPORT, "part2_group_detected.png"), out)

    GT = 10
    print("\n=== Part 2.2: group photo detection ===")
    print(f"  ground truth faces: {GT}  (8 detectable + 2 expected-hard: 90deg-sideways, 85%-occluded)")
    print(f"  detected:           {len(boxes)}")
    print(f"  confidences:        {', '.join(f'{b[4]:.2f}' for b in boxes)}")
    print(f"  saved -> {os.path.join(REPORT, 'part2_group_detected.png')}")
    return GT, boxes


if __name__ == "__main__":
    demo_database_images()
    build_group_collage()
