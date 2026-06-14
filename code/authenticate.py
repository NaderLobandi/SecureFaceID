"""
Project 6 - Part 6: live face authentication app (Face ID-style GUI).

Pipeline per frame: capture -> YOLO detect -> (bg) embed+match -> (bg) attributes
-> decide AUTHENTICATED / DENIED, drawn on a dark iOS-style canvas with an
animated scan ring, a morphing lock icon, a confidence arc, and check/shake
feedback. Heavy work (embedding, DeepFace, Claude vision) runs on background
threads so the video stays smooth.

Run from the project root:
    python code/authenticate.py            # uses ME below
    python code/authenticate.py --me me --threshold 0.42

Keys: ESC = quit,  S = save a screenshot to report/.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import math
import time
import pickle
import argparse
import threading
import warnings
import cv2
import numpy as np

warnings.filterwarnings("ignore")
from detect import detect_faces
from embed import embed_face, identify, PKL_PATH
from describe import describe_face, describe_face_with_genai

# ---- config ----
ME = "me"
THRESHOLD = 0.42            # operating point (Part 5 finding; Part 6 study confirms)
EXIT_THRESHOLD = 0.38      # hysteresis: stay authenticated down to here
CONFIRM_FRAMES = 4         # consecutive frames needed to flip state
PANEL_W = 360
DARK = (28, 26, 24)        # BGR near-black
GREEN = (90, 220, 120)
RED = (70, 70, 235)
GREY = (150, 150, 150)
WHITE = (245, 245, 245)
REPORT = os.path.join(os.path.dirname(__file__), "..", "report")


# ---------- shared state between threads ----------
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.crop = None            # latest face crop for workers
        self.name = "Unknown"
        self.score = 0.0            # EMA-smoothed similarity
        self.attrs = {"age": "-", "gender": "-", "expression": "-", "glasses": "-",
                      "facial_hair": "-", "mask": "-", "eyes_open": "-",
                      "image_quality": "-"}
        self.rich = ""
        self.running = True

    def set_crop(self, crop):
        with self.lock:
            self.crop = crop

    def get_crop(self):
        with self.lock:
            return None if self.crop is None else self.crop.copy()


# ---------- background workers ----------
def match_worker(state, db):
    """Embed the latest crop and update the smoothed best-match score."""
    while state.running:
        crop = state.get_crop()
        if crop is None or crop.size == 0:
            time.sleep(0.03); continue
        try:
            emb = embed_face(crop)
            name, score = identify(emb, db, threshold=-1.0)   # always best match
            with state.lock:
                state.name = name
                state.score = 0.5 * state.score + 0.5 * score   # EMA
        except Exception:
            pass


def attr_worker(state):
    """Fast local attributes (age/gender/expression/quality) via DeepFace."""
    while state.running:
        crop = state.get_crop()
        if crop is None or crop.size == 0:
            time.sleep(0.05); continue
        a = describe_face(crop, use_genai=False)
        with state.lock:
            for k in ("age", "gender", "expression", "image_quality"):
                state.attrs[k] = a[k]


def vision_worker(state):
    """Claude vision: glasses/facial_hair/mask/eyes_open + rich desc, ~every 2s."""
    from describe import vision_appearance
    while state.running:
        crop = state.get_crop()
        if crop is None or crop.size == 0:
            time.sleep(0.2); continue
        app = vision_appearance(crop)
        if app is not None:
            with state.lock:
                state.attrs["glasses"] = "Yes" if app.glasses else "No"
                state.attrs["facial_hair"] = app.facial_hair
                state.attrs["mask"] = "Yes" if app.mask else "No"
                state.attrs["eyes_open"] = "Yes" if app.eyes_open else "No"
        rich = describe_face_with_genai(crop)
        if rich:
            with state.lock:
                state.rich = rich
        time.sleep(2.0)


# ---------- drawing helpers ----------
def rounded_rect(img, p1, p2, color, thickness=-1, r=12):
    x1, y1 = p1; x2, y2 = p2
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        for cx, cy, a0, a1 in ((x1+r, y1+r, 180, 270), (x2-r, y1+r, 270, 360),
                               (x1+r, y2-r, 90, 180), (x2-r, y2-r, 0, 90)):
            cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a1, color, thickness)


def draw_scan_ring(img, center, radius, phase, color, n=28):
    """Rotating dashed ring (Face ID scan sweep)."""
    for i in range(n):
        ang = math.radians(i * 360 / n)
        # brightness sweeps around the ring
        t = (math.sin(math.radians(i * 360 / n) - phase) + 1) / 2
        c = tuple(int(40 + (v - 40) * t) for v in color)
        x1 = int(center[0] + radius * math.cos(ang))
        y1 = int(center[1] + radius * math.sin(ang))
        x2 = int(center[0] + (radius + 9) * math.cos(ang))
        y2 = int(center[1] + (radius + 9) * math.sin(ang))
        cv2.line(img, (x1, y1), (x2, y2), c, 2)


def draw_confidence_arc(img, center, radius, score, color):
    cv2.ellipse(img, center, (radius, radius), 0, -90, -90 + 360 * max(0, min(1, score)),
                color, 4, cv2.LINE_AA)


def draw_lock(img, center, unlocked, color):
    cx, cy = center
    rounded_rect(img, (cx - 13, cy), (cx + 13, cy + 20), color, -1, r=4)  # body
    if unlocked:                                                          # open shackle
        cv2.ellipse(img, (cx + 6, cy - 4), (9, 11), 0, 150, 360, color, 3)
    else:                                                                 # closed shackle
        cv2.ellipse(img, (cx, cy - 4), (9, 11), 0, 180, 360, color, 3)


def draw_check(img, center, t, color):
    """Animated checkmark, t in [0,1] scales it in."""
    cx, cy = center; s = int(20 * t)
    if s < 2:
        return
    cv2.line(img, (cx - s, cy), (cx - s // 3, cy + s), color, 4, cv2.LINE_AA)
    cv2.line(img, (cx - s // 3, cy + s), (cx + s, cy - s), color, 4, cv2.LINE_AA)


def wrap(text, width=34):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def draw_panel(min_h, state, status, ok, fps, check_t, live_info=(False, 0.0, True, 0)):
    with state.lock:
        name, score = state.name, state.score
        attrs = dict(state.attrs); rich = state.rich

    rows = [("Age", attrs["age"]), ("Gender", attrs["gender"]),
            ("Expression", attrs["expression"]), ("Glasses", attrs["glasses"]),
            ("Facial hair", attrs["facial_hair"]), ("Mask", attrs["mask"]),
            ("Eyes open", attrs["eyes_open"]), ("Quality", attrs["image_quality"])]
    live_on, ear, blinked, blink_count = live_info
    if live_on:
        rows.append(("Liveness", f"live ({blink_count})" if blinked else "blink to verify"))
        rows.append(("EAR", f"{ear:.2f}"))
    rows.append(("FPS", f"{fps:.1f}"))

    rich_lines = wrap(rich or "(rich description loading...)")[:4]
    needed = 104 + len(rows) * 36 + 24 + len(rich_lines) * 18 + 16
    h = max(min_h, needed)
    panel = np.full((h, PANEL_W, 3), DARK, np.uint8)
    accent = (GREEN if ok else RED if status == "DENIED"
              else (40, 170, 255) if status == "LIVENESS" else GREY)

    # status pill
    rounded_rect(panel, (16, 16), (PANEL_W - 16, 58), accent, -1, r=20)
    label = "BLINK NEEDED" if status == "LIVENESS" else status
    cv2.putText(panel, label, (30, 46), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 20, 20), 2)

    ident = f"{name}  sim={score:.2f}" if ok else (
        f"not {ME}  (best {name} {score:.2f})" if status == "DENIED" else "scanning...")
    cv2.putText(panel, ident, (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    # attribute cards
    y = 104
    for lbl, val in rows:
        rounded_rect(panel, (16, y), (PANEL_W - 16, y + 30), (44, 42, 40), -1, r=8)
        cv2.putText(panel, lbl, (26, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1)
        cv2.putText(panel, str(val), (160, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        y += 36

    # rich description strip
    cv2.line(panel, (16, y + 4), (PANEL_W - 16, y + 4), (60, 58, 56), 1)
    y += 24
    for line in rich_lines:
        cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        y += 18

    # animated check on success
    if ok and check_t < 1.0:
        draw_check(panel, (PANEL_W - 44, 37), check_t, (20, 20, 20))
    return panel


# ---------- main loop ----------
def main(me, threshold, cam_index=0, liveness=False):
    global ME, THRESHOLD
    ME, THRESHOLD = me, threshold
    db = pickle.load(open(PKL_PATH, "rb"))
    print(f"loaded {len(db)} identities; ME={ME}, threshold={THRESHOLD}, "
          f"liveness={'ON' if liveness else 'OFF'}")

    detector = None
    if liveness:
        from liveness import BlinkDetector
        detector = BlinkDetector()
        print("liveness ON: a blink is required before authenticating.")

    state = State()
    workers = [threading.Thread(target=match_worker, args=(state, db), daemon=True),
               threading.Thread(target=attr_worker, args=(state,), daemon=True),
               threading.Thread(target=vision_worker, args=(state,), daemon=True)]
    for w in workers:
        w.start()

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        state.running = False
        raise SystemExit("Could not open webcam (grant camera permission on macOS).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    status = "SCANNING"
    me_streak = other_streak = 0
    check_t = 1.0
    shake = 0
    t0, frames, shot_n = time.time(), 0, 0
    print("ESC = quit, S = screenshot")

    while True:
        ok_cap, frame = cap.read()
        if not ok_cap:
            break
        # NOTE: do NOT mirror - enrollment frames were saved un-flipped, and a
        # horizontal flip lowers match similarity (our genuine/impostor margin
        # is tight). Keep orientation consistent with the gallery.
        frames += 1
        fps = frames / (time.time() - t0 + 1e-6)

        faces = detect_faces(frame, conf=0.4)
        box = max(faces, key=lambda b: (b[2]-b[0])*(b[3]-b[1])) if faces else None
        if box:
            x1, y1, x2, y2, _ = box
            state.set_crop(frame[y1:y2, x1:x2])
        else:
            state.set_crop(None)

        with state.lock:
            name, score = state.name, state.score

        # ---- liveness (blink) check ----
        now = time.time()
        ear, blinked = 0.0, True               # blinked=True when liveness off
        if detector is not None:
            ear = detector.update(frame, now) if box else 0.0
            blinked = detector.blinked_recently(now)

        # ---- hysteresis state machine ----
        if box and name == ME and score >= THRESHOLD:
            me_streak += 1; other_streak = 0
        elif box and (name != ME or score < EXIT_THRESHOLD):
            other_streak += 1; me_streak = 0
        else:
            me_streak = max(0, me_streak - 1)   # holding zone
        prev = status
        if me_streak >= CONFIRM_FRAMES:
            # matched - but require a recent blink when liveness is on
            status = "AUTHENTICATED" if blinked else "LIVENESS"
        elif other_streak >= CONFIRM_FRAMES:
            status = "DENIED"
        elif not box:
            status = "SCANNING"
        if status == "AUTHENTICATED" and prev != "AUTHENTICATED":
            check_t = 0.0                       # trigger check-in animation
        if status == "DENIED" and prev != "DENIED":
            shake = 12                          # trigger shake
        ok = status == "AUTHENTICATED"
        check_t = min(1.0, check_t + 0.12)
        shake = max(0, shake - 1)

        # ---- draw on video ----
        ORANGE = (40, 170, 255)
        accent = (GREEN if ok else RED if status == "DENIED"
                  else ORANGE if status == "LIVENESS" else (200, 200, 80))
        if box:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = int(max(x2 - x1, y2 - y1) * 0.62)
            rounded_rect(frame, (x1, y1), (x2, y2), accent, 2, r=14)
            if status == "SCANNING":
                draw_scan_ring(frame, (cx, cy), radius, frames * 0.25, (200, 200, 90))
            else:
                draw_confidence_arc(frame, (cx, cy), radius, min(score, 1.0), accent)
                draw_lock(frame, (cx, y1 - 30), ok, accent)
                if ok and check_t < 1.0:
                    draw_check(frame, (cx, cy), check_t, GREEN)
            if status == "LIVENESS":
                cv2.putText(frame, "BLINK to verify", (x1, y2 + 28),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, ORANGE, 2)

        live_info = (detector is not None, ear, blinked,
                     detector.blink_count if detector else 0)
        panel = draw_panel(frame.shape[0], state, status, ok, fps, check_t, live_info)
        if shake > 0:                           # nudge the panel on denial
            panel = np.roll(panel, int(6 * math.sin(shake)), axis=1)
        # pad video to panel height so hstack aligns (panel can be taller)
        if panel.shape[0] > frame.shape[0]:
            pad = np.full((panel.shape[0] - frame.shape[0], frame.shape[1], 3), DARK, np.uint8)
            frame = np.vstack([frame, pad])
        out = np.hstack([frame, panel])

        cv2.imshow("Face Authentication - ESC to quit", out)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k in (ord("s"), ord("S")):
            shot_n += 1
            p = os.path.join(REPORT, f"gui_screenshot_{status.lower()}_{shot_n}.png")
            cv2.imwrite(p, out); print("saved", p)

    state.running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", default=ME)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--liveness", action="store_true",
                    help="require a blink (EAR) before authenticating (anti-spoofing)")
    args = ap.parse_args()
    main(args.me.lower(), args.threshold, cam_index=args.cam, liveness=args.liveness)
