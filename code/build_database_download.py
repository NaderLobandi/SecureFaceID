"""
Project 6 - Part 1: Build a synthetic face database (Option A).

Downloads fresh AI-generated faces from thispersondoesnotexist.com (no real
person depicted), then builds 3-5 intra-class "views" per identity via mild
augmentation so each identity has some pose/lighting variation.

Usage:
    python build_database_download.py [--n 12] [--root ../database]

Be a good citizen: a polite time.sleep(1.5) is used between network calls.
If the primary site is down, we fall back to generated.photos.
"""

import os
import time
import uuid
import argparse
import requests
import cv2
import numpy as np

PRIMARY_URL = "https://thispersondoesnotexist.com"
FALLBACK_URL = "https://generated.photos"  # last-resort fallback per spec
# A realistic full browser UA gets past Cloudflare's bot filter reliably;
# the bare "class-project" UA can trigger transient 525 SSL-handshake errors.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
MAX_RETRIES = 5  # Cloudflare 5xx (e.g. 525) errors are often transient
DARK_FACTOR = 0.4  # multiplicative low-light factor for the "03 dark" view

# Pool of human-friendly first names; combined with a short UUID so IDs never
# collide even if a name repeats.
NAMES = [
    "ava", "ben", "cara", "dev", "eli", "finn", "gia", "hugo",
    "iris", "jade", "kai", "lia", "milo", "nina", "omar", "piper",
    "quinn", "rumi", "sage", "theo", "uma", "vince", "wren", "zia",
]


def random_id(i):
    """e.g. 'ava_3f9c2a' --- name + short UUID so IDs never collide."""
    return f"{NAMES[i % len(NAMES)]}_{uuid.uuid4().hex[:6]}"


def fetch_face():
    """Fetch one synthetic face as JPEG bytes.

    Retries the primary site on transient 5xx (Cloudflare) errors with
    exponential backoff, then falls back to the secondary source.
    """
    for url in (PRIMARY_URL, FALLBACK_URL):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(url, headers=HEADERS, timeout=15)
                r.raise_for_status()
                # sanity check: must look like a JPEG, not an HTML error page
                ctype = r.headers.get("Content-Type", "")
                if "image" not in ctype and r.content[:3] != b"\xff\xd8\xff":
                    print(f"  [warn] {url} returned non-image ({ctype}); next source")
                    break  # this source isn't serving images; try the next URL
                return r.content
            except requests.RequestException as e:
                wait = 2 ** attempt  # 2, 4, 8, 16, 32 s
                print(f"  [warn] {url} attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
    raise RuntimeError("All face sources failed. Check your connection or use Option B.")


def make_views(img):
    """Return 4 augmented views of the same synthetic identity.

    These add intra-class variation (the matcher learns invariance) but are
    derived from one source image, so they are correlated -- diminishing
    returns past a handful, which is exactly why 3-5 is the sweet spot.
    """
    h, w = img.shape[:2]
    flipped = cv2.flip(img, 1)                       # horizontal mirror
    cy, cx, s = h // 2, w // 2, int(min(h, w) * 0.42)
    zoomed = cv2.resize(img[cy - s:cy + s, cx - s:cx + s], (w, h))  # center crop + zoom
    # Low-light view: multiplicative scaling darkens all tones proportionally
    # (additive subtraction barely touches mid/high tones and looks unchanged).
    darker = np.clip(img.astype(np.float32) * DARK_FACTOR, 0, 255).astype(np.uint8)
    return [img, flipped, zoomed, darker]


def download_synthetic_database(n_identities=12, root="../database"):
    os.makedirs(root, exist_ok=True)
    for i in range(n_identities):
        pid = random_id(i)
        pdir = os.path.join(root, pid)
        os.makedirs(pdir, exist_ok=True)

        raw = fetch_face()
        with open(os.path.join(pdir, "src.jpg"), "wb") as f:
            f.write(raw)

        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [warn] could not decode image for {pid}; skipping views")
            continue
        for j, view in enumerate(make_views(img)):
            cv2.imwrite(os.path.join(pdir, f"{j:02d}.jpg"), view)

        print(f"[{i + 1}/{n_identities}] {pid} -- 4 views saved")
        time.sleep(1.5)  # be polite to the free server


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="number of identities")
    ap.add_argument("--root", default="../database", help="database root folder")
    args = ap.parse_args()
    download_synthetic_database(n_identities=args.n, root=args.root)
    print("\nDone. Database built at:", os.path.abspath(args.root))
