"""
Project 6 - Part 4: face attribute recognition.

describe_face(bgr_crop) returns the four required attributes
{age, gender, expression, glasses} PLUS real-system extras
{facial_hair, mask, eyes_open, image_quality}.

Method split (one paid API call total):
  - age / gender / expression : DeepFace.analyze (local)
  - glasses / facial_hair / mask / eyes_open : ONE Claude vision call,
    returned as a validated structured object (no fragile text parsing)
  - image_quality : local OpenCV (sharpness + brightness)

describe_face_with_genai(bgr_crop) is the optional Route-C rich one-sentence
description for the Part 6 side panel.

Everything degrades gracefully: with no ANTHROPIC_API_KEY (or on any API error)
the appearance attrs fall back to safe defaults so the app never crashes offline.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import base64
import warnings
import cv2

warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from deepface import DeepFace

import anthropic
from pydantic import BaseModel, Field

# Current Claude model (spec's claude-opus-4-7 is outdated). Vision-capable.
VISION_MODEL = "claude-opus-4-8"

# Exact prompt sent to Claude (also logged in ai_usage.md).
VISION_PROMPT = (
    "You are analyzing a single cropped face image for a face-attribute demo. "
    "Report only what is visibly true in the image. "
    "glasses: true if the person is wearing eyeglasses or sunglasses. "
    "facial_hair: one of 'none', 'stubble', 'beard', 'mustache'. "
    "mask: true if a face mask or covering obscures the nose or mouth. "
    "eyes_open: true if both eyes are open. "
    "Do NOT identify, name, or guess the identity of the person."
)

RICH_PROMPT = (
    "In ONE sentence, describe the hair, accessories, and visible mood of this "
    "face. Do NOT attempt to identify the person."
)


class Appearance(BaseModel):
    """Structured schema the vision model is forced to return."""
    glasses: bool = Field(description="wearing eyeglasses or sunglasses")
    facial_hair: str = Field(description="none, stubble, beard, or mustache")
    mask: bool = Field(description="a mask/covering obscures nose or mouth")
    eyes_open: bool = Field(description="both eyes are open")


_client = None


def _get_client():
    """Return a cached Anthropic client, or None if no API key is configured."""
    global _client
    if _client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        _client = anthropic.Anthropic()
    return _client


def _encode_jpeg_b64(bgr_crop):
    ok, buf = cv2.imencode(".jpg", bgr_crop)
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def assess_quality(bgr_crop):
    """Local capture-quality gate: classify by brightness then sharpness."""
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if brightness < 60:
        return "low-light"
    if brightness > 210:
        return "overexposed"
    if sharpness < 40:
        return "blurry"
    return "good"


def vision_appearance(bgr_crop):
    """One Claude vision call -> validated Appearance, or None on failure."""
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.messages.parse(
            model=VISION_MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": _encode_jpeg_b64(bgr_crop)}},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }],
            output_format=Appearance,
        )
        if resp.stop_reason == "refusal":
            return None
        return resp.parsed_output
    except Exception as e:
        print(f"  [warn] vision_appearance failed: {e}")
        return None


def describe_face(bgr_crop, use_genai=True):
    """Return the 4 required attrs + extras as a flat dict (Part 6 reads this)."""
    out = {
        # --- required ---
        "age": None, "gender": "unknown", "expression": "unknown", "glasses": False,
        # --- augmented (real-system) ---
        "facial_hair": "unknown", "mask": False, "eyes_open": True,
        "image_quality": "unknown",
    }

    # age / gender / expression via DeepFace (local)
    try:
        a = DeepFace.analyze(
            bgr_crop, actions=["age", "gender", "emotion"],
            enforce_detection=False, detector_backend="skip")[0]
        out["age"] = int(a["age"])
        out["gender"] = a["dominant_gender"]
        out["expression"] = a["dominant_emotion"]
    except Exception as e:
        print(f"  [warn] DeepFace.analyze failed: {e}")

    # image quality (local)
    out["image_quality"] = assess_quality(bgr_crop)

    # glasses / facial_hair / mask / eyes_open via one Claude vision call
    if use_genai:
        app = vision_appearance(bgr_crop)
        if app is not None:
            out["glasses"] = app.glasses
            out["facial_hair"] = app.facial_hair
            out["mask"] = app.mask
            out["eyes_open"] = app.eyes_open

    return out


def describe_face_with_genai(bgr_crop):
    """Route-C bonus: one-sentence rich description, or None if unavailable."""
    client = _get_client()
    if client is None:
        return None
    try:
        msg = client.messages.create(
            model=VISION_MODEL,
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": _encode_jpeg_b64(bgr_crop)}},
                    {"type": "text", "text": RICH_PROMPT},
                ],
            }],
        )
        if msg.stop_reason == "refusal":
            return None
        return next((b.text for b in msg.content if b.type == "text"), None)
    except Exception as e:
        print(f"  [warn] describe_face_with_genai failed: {e}")
        return None


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1
                     else "../database/ava_48c7d6/00.jpg")
    print("attributes:", describe_face(img))
    print("rich:", describe_face_with_genai(img))
