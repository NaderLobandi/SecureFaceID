"""
Project 6 - Part 5: enroll yourself into the database from the webcam.

Opens the webcam, draws a live YOLO box so you can frame yourself, and saves
15 frames to database/<name>/ when you press SPACE. An on-screen hint walks you
through the required pose/lighting variation.

Usage (from the project root):
    python code/enroll.py --name me

Keys:  SPACE = capture frame   ESC = quit
macOS: grant camera permission to your terminal/VSCode in
       System Settings > Privacy & Security > Camera.
"""

import os
import argparse
import cv2
from detect import detect_faces

N_FRAMES = 15
# Per-frame guidance (index -> hint). Covers the spec's required variation.
HINTS = [
    "front view, neutral", "front view, smile", "front view, eyes wide",
    "turn head slightly LEFT", "turn slightly LEFT (more)", "turn head slightly RIGHT",
    "turn slightly RIGHT (more)", "tilt head UP a little", "tilt UP (more)",
    "tilt head DOWN a little", "tilt DOWN (more)", "glasses ON (or neutral)",
    "glasses OFF (or smile)", "move to brighter light", "move to dimmer light",
]


def main(name, cam_index=0):
    out_dir = os.path.join(os.path.dirname(__file__), "..", "database", name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(
            "Could not open the webcam. On macOS, grant camera permission to "
            "your terminal/VSCode (System Settings > Privacy & Security > Camera).")
    # 640x480 keeps detection fast and is plenty for embeddings.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count = 0
    print("SPACE = capture, ESC = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        display = frame.copy()

        # live YOLO box so you can frame yourself well
        faces = detect_faces(frame, conf=0.4)
        for (x1, y1, x2, y2, _) in faces:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        hint = HINTS[count] if count < N_FRAMES else "done - press ESC"
        cv2.putText(display, f"captured: {count}/{N_FRAMES}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"NEXT: {hint}", (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.putText(display, "SPACE = capture   ESC = quit", (12, display.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Enroll yourself", display)

        k = cv2.waitKey(1) & 0xFF
        if k == 32 and count < N_FRAMES:           # SPACE
            if not faces:
                print(f"  [skip] no face detected - reframe and try again")
                continue
            path = os.path.join(out_dir, f"{count:02d}.jpg")
            cv2.imwrite(path, frame)               # save the clean frame (no overlay)
            print(f"  captured {count + 1}/{N_FRAMES} -> {path}")
            count += 1
            if count == N_FRAMES:
                print("All 15 captured. Press ESC to finish.")
        elif k == 27:                              # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {count} frame(s) saved to {os.path.abspath(out_dir)}")
    if count < N_FRAMES:
        print(f"NOTE: only {count}/{N_FRAMES} captured - re-run to add more.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="your lowercase folder name, e.g. me")
    ap.add_argument("--cam", type=int, default=0, help="webcam index (default 0)")
    args = ap.parse_args()
    main(args.name.lower(), cam_index=args.cam)
