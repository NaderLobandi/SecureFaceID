"""
Project 6 - Part 3: face embeddings, database building, and matching.

Shared module imported by Part 6. YOLO (detect.py) provides the crop; DeepFace
provides ONLY the embedding (detector_backend='skip' since we already cropped).

Cosine similarity is the distance metric; all embeddings are L2-normalised so a
plain dot product equals cosine similarity.
"""

import os
# Quiet TensorFlow before it is imported anywhere.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pickle
import warnings
import cv2
import numpy as np

warnings.filterwarnings("ignore")
from deepface import DeepFace
from detect import detect_faces

# Production backbone for the live app (Parts 5/6). Chosen in Part 3: VGG-Face
# had the largest worst-case held-out margin on our data. DeepFace also supports
# 'Facenet512', 'ArcFace', 'Facenet'.
EMBED_MODEL = "VGG-Face"

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_ROOT = os.path.join(_HERE, "..", "database")
PKL_PATH = os.path.join(_HERE, "..", "face_db.pkl")


def embed_face(bgr_crop, model_name=EMBED_MODEL):
    """Return a single unit-norm embedding from a face crop."""
    rep = DeepFace.represent(
        bgr_crop,
        model_name=model_name,
        enforce_detection=False,   # YOLO already cropped
        detector_backend="skip",
    )
    v = np.array(rep[0]["embedding"], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def largest_face_crop(bgr_image, conf=0.4):
    """Detect faces and return the crop of the largest one, or None."""
    faces = detect_faces(bgr_image, conf=conf)
    if not faces:
        return None
    x1, y1, x2, y2, _ = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return bgr_image[y1:y2, x1:x2]


def build_database(root=DB_ROOT, model_name=EMBED_MODEL, save_path=PKL_PATH,
                   exclude_src=True, verbose=True):
    """Build {name: (N, d) stacked embeddings} and optionally pickle it.

    exclude_src=True skips 'src.jpg' (a byte-for-byte duplicate of 00.jpg) so an
    identical copy does not leak into the gallery and inflate accuracy.
    """
    db = {}
    for name in sorted(os.listdir(root)):
        person_dir = os.path.join(root, name)
        if not os.path.isdir(person_dir):
            continue
        embs = []
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if exclude_src and fname == "src.jpg":
                continue
            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                continue
            crop = largest_face_crop(img)
            if crop is None or crop.size == 0:
                continue
            embs.append(embed_face(crop, model_name=model_name))
        if embs:
            db[name] = np.stack(embs)
            if verbose:
                print(f"  {name}: {len(embs)} embeddings")
    if save_path:
        pickle.dump(db, open(save_path, "wb"))
        if verbose:
            print(f"saved -> {save_path}")
    return db


def identify(query_emb, db, threshold=0.55):
    """Return (best_name, best_score). 'Unknown' if no match passes threshold."""
    best_name, best_score = "Unknown", -1.0
    for name, embs in db.items():
        # cosine sim == dot product because everything is unit-norm
        s = float((embs @ query_emb).max())
        if s > best_score:
            best_score, best_name = s, name
    if best_score < threshold:
        return "Unknown", best_score
    return best_name, best_score


if __name__ == "__main__":
    print(f"Building face database with {EMBED_MODEL} ...")
    db = build_database()
    n_emb = sum(len(v) for v in db.values())
    print(f"\n{len(db)} identities, {n_emb} total embeddings.")
