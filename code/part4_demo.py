"""
Project 6 - Part 4 deliverable: attribute comparison table over several faces.

For each chosen face: YOLO-crop the largest face, run describe_face(), and render
a thumbnail + the eight attributes. Saves a figure and a markdown table.

Spec asks for 4 faces (3 DB + 1 of you). The "you" face is added in Part 5;
for now this runs over DB faces. Re-run after enrollment to include yourself.

Outputs: ../report/part4_attributes.png, ../report/part4_results.md
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import glob
import warnings
import cv2
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detect import detect_faces
from describe import describe_face, describe_face_with_genai, VISION_MODEL

REPORT = os.path.join(os.path.dirname(__file__), "..", "report")
DB = os.path.join(os.path.dirname(__file__), "..", "database")
KEYS = ["age", "gender", "expression", "glasses",
        "facial_hair", "mask", "eyes_open", "image_quality"]


def largest_crop(img):
    faces = detect_faces(img, conf=0.4)
    if not faces:
        return img
    x1, y1, x2, y2, _ = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return img[y1:y2, x1:x2]


def main(extra_paths=None, use_genai=True):
    # 3 database faces (+ optional extras, e.g. your enrolled face from Part 5)
    ids = sorted(glob.glob(os.path.join(DB, "*", "")))
    paths = [os.path.join(ids[i], "00.jpg") for i in (3, 8, 14)]
    labels = [os.path.basename(ids[i].rstrip("/")) for i in (3, 8, 14)]
    for p in (extra_paths or []):
        paths.append(p); labels.append(os.path.basename(os.path.dirname(p)) or "you")

    results = []
    for path, label in zip(paths, labels):
        img = cv2.imread(path)
        if img is None:
            continue
        crop = largest_crop(img)
        attrs = describe_face(crop, use_genai=use_genai)
        rich = describe_face_with_genai(crop) if use_genai else None
        results.append((label, crop, attrs, rich))
        print(f"{label}: {attrs}")
        if rich:
            print(f"    rich: {rich}")

    _figure(results)
    _markdown(results, use_genai)


def _figure(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, (label, crop, attrs, _) in zip(axes, results):
        ax.imshow(cv2.cvtColor(cv2.resize(crop, (200, 200)), cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=9)
        ax.axis("off")
        txt = "\n".join(f"{k}: {attrs[k]}" for k in KEYS)
        ax.text(0.5, -0.04, txt, transform=ax.transAxes, fontsize=8,
                va="top", ha="center", family="monospace")
    fig.suptitle("Part 4 — describe_face() attributes", fontsize=12)
    fig.tight_layout()
    path = os.path.join(REPORT, "part4_attributes.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("figure ->", path)


def _markdown(results, use_genai):
    lines = [
        "# Part 4 — Face Attribute Recognition\n",
        "`describe_face(bgr_crop)` returns the 4 required attributes "
        "(age, gender, expression, glasses) plus real-system extras "
        "(facial_hair, mask, eyes_open, image_quality).\n",
        "**Method:** age/gender/expression from DeepFace.analyze (local); "
        "glasses/facial_hair/mask/eyes_open from ONE Claude vision call "
        f"(`{VISION_MODEL}`, structured output); image_quality from local "
        "OpenCV (Laplacian sharpness + brightness).\n",
        "**Glasses route:** B.3 — Vision LLM (Claude). One structured call also "
        "yields facial_hair/mask/eyes_open, so all four appearance attrs cost a "
        "single API request. Exact prompt is logged in `ai_usage.md`.\n",
        "**Ethics note:** race/ethnicity is deliberately excluded — DeepFace can "
        "report it but it is biased and inappropriate for authentication.\n",
        "| Face | " + " | ".join(KEYS) + " |",
        "|" + "---|" * (len(KEYS) + 1),
    ]
    for label, _, attrs, _ in results:
        lines.append(f"| {label} | " + " | ".join(str(attrs[k]) for k in KEYS) + " |")
    if use_genai:
        lines += ["", "## Rich descriptions (Route C bonus)"]
        for label, _, _, rich in results:
            lines.append(f"- **{label}:** {rich or '_(unavailable)_'}")
    path = os.path.join(REPORT, "part4_results.md")
    open(path, "w").write("\n".join(lines))
    print("report ->", path)


if __name__ == "__main__":
    import sys
    main(use_genai="--no-genai" not in sys.argv)
