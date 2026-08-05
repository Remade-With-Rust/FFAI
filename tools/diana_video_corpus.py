"""Build Diana's video corpus, mirroring how Ultralytics ingests video.

Ultralytics reads video with `cv2.VideoCapture` and hands each decoded frame
to the model. This extracts frames the same way, so BOTH engines are fed the
identical decoded pixels and neither pays for a decode the other does not —
the work-parity rule that voided an earlier comparison in this campaign when
our harness pre-decoded and the reference did not.

The clips are SYNTHESISED rather than downloaded, deliberately:

* they must be redistributable, and a hash-pinned corpus of someone else's
  footage is not;
* the question the LIVE gate needs answered is about CODEC NOISE, and that is
  a property of the encoder, not of the scene;
* the motion has to be ground-truth known, because the whole point is to
  separate "the picture changed" from "the encoder wobbled".

Each clip is encoded with a real codec and decoded back, so the frames carry
genuine quantisation artefacts.
"""

import argparse
import hashlib
import json
import os
import shutil

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(ROOT, "corpora", "clips", "diana-video")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def scenes(src, w, h, n):
    """(name, frames, motion_frame_indices) for each scene class."""
    out = []

    # STATIC: a fixed camera on an unchanging scene. Every inter-frame
    # difference here is the encoder, not the world.
    out.append(("static", [src.copy() for _ in range(n)], set()))

    # WALK: an object crossing the frame — the motion a detector must not
    # miss. Sized like a real subject, not a token box.
    fr, motion = [], set()
    ow, oh = w // 5, h // 3
    for i in range(n):
        f = src.copy()
        if i >= n // 3:
            x = int((i - n // 3) * (w - ow) / (n - n // 3 - 1))
            cv2.rectangle(f, (x, h - oh - 10), (x + ow, h - 10), (40, 190, 60), -1)
            motion.add(i)
        fr.append(f)
    out.append(("walk", fr, motion))

    # PAN: the camera itself moves, one pixel per frame. The worst case for a
    # change gate and the one that decides whether it helps handheld footage.
    fr = [np.roll(src, i, axis=1) for i in range(n)]
    out.append(("pan", fr, set(range(1, n))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fourcc", default="avc1", help="avc1 (H.264) or mp4v")
    ap.add_argument("--source", default="corpora/clips/diana-coco/coco-032.png")
    args = ap.parse_args()

    src = cv2.imread(os.path.join(ROOT, args.source))
    if src is None:
        raise SystemExit(f"cannot read {args.source}")
    h, w = src.shape[:2]

    if os.path.isdir(CLIPS):
        shutil.rmtree(CLIPS)
    os.makedirs(CLIPS)

    manifest = {"source": args.source, "fourcc": args.fourcc, "clips": []}
    for name, frames, motion in scenes(src, w, h, args.frames):
        d = os.path.join(CLIPS, name)
        os.makedirs(d)
        mp4 = os.path.join(d, f"{name}.mp4")
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*args.fourcc), 30, (w, h))
        if not vw.isOpened():
            raise SystemExit(f"VideoWriter failed for {args.fourcc}")
        for f in frames:
            vw.write(f)
        vw.release()

        # Decode back — THESE are the pixels both engines see.
        cap = cv2.VideoCapture(mp4)
        got = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            cv2.imwrite(os.path.join(d, f"f{got:04d}.png"), f)
            got += 1
        cap.release()

        manifest["clips"].append(
            {
                "name": name,
                "video": os.path.relpath(mp4, ROOT).replace("\\", "/"),
                "video_sha256": sha256(mp4),
                "frames": got,
                "width": w,
                "height": h,
                # Ground truth: which frames genuinely differ from their
                # predecessor. Everything else is encoder noise.
                "motion_frames": sorted(motion),
            }
        )
        print(f"{name:8s} {got:3d} frames, {os.path.getsize(mp4)//1024:5d} KiB, "
              f"{len(motion):3d} with real motion")

    mpath = os.path.join(CLIPS, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nwrote {os.path.relpath(mpath, ROOT)}")


if __name__ == "__main__":
    main()
