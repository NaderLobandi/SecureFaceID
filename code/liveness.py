"""
Project 6 - Graduate anti-spoofing: blink-based liveness via Eye Aspect Ratio.

Uses MediaPipe FaceLandmarker (Tasks API, 478 landmarks) to compute EAR for both
eyes. A live person blinks (EAR dips below ~0.20 then recovers); a printed photo
never does. BlinkDetector tracks a close->open transition and reports whether a
full blink happened within the last few seconds.

EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)   per eye (6 landmarks).

Model: code/models/face_landmarker.task (MediaPipe, ~3.6MB).
"""

import os
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(_HERE, "models", "face_landmarker.task")

# 6-point EAR landmark indices on the MediaPipe face mesh.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_CLOSED = 0.20    # below this = eye closed
EAR_OPEN = 0.25      # above this (after a close) = blink completed
BLINK_WINDOW_S = 3.0  # a blink counts as "recent" for this long


def _ear(pts, idx, w, h):
    p = [(pts[i].x * w, pts[i].y * h) for i in idx]
    d = lambda a, b: math.dist(a, b)
    horiz = d(p[0], p[3])
    if horiz < 1e-6:
        return 0.0
    return (d(p[1], p[5]) + d(p[2], p[4])) / (2 * horiz)


class BlinkDetector:
    def __init__(self):
        opts = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=MODEL),
            num_faces=1, running_mode=vision.RunningMode.IMAGE)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.ear = 0.0
        self.eye_closed = False
        self.last_blink_t = -1e9
        self.blink_count = 0

    def update(self, bgr_frame, now):
        """Run one frame; update EAR and blink state. Returns current EAR."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(mimg)
        if not res.face_landmarks:
            self.ear = 0.0
            return 0.0
        pts = res.face_landmarks[0]
        h, w = bgr_frame.shape[:2]
        self.ear = (_ear(pts, LEFT_EYE, w, h) + _ear(pts, RIGHT_EYE, w, h)) / 2

        # blink = closed then reopened
        if self.ear < EAR_CLOSED:
            self.eye_closed = True
        elif self.ear > EAR_OPEN and self.eye_closed:
            self.eye_closed = False
            self.last_blink_t = now
            self.blink_count += 1
        return self.ear

    def blinked_recently(self, now, window=BLINK_WINDOW_S):
        return (now - self.last_blink_t) <= window
