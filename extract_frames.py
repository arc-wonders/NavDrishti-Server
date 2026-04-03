#!/usr/bin/env python3
"""Dump frames from a specific video using hard-coded paths/settings."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2

VIDEO_PATH = Path("video.mp4")
OUTPUT_DIR = Path("frames")
EVERY_N_FRAMES = 300
FRAME_LIMIT = 900  # Set to an int to stop after saving that many frames (None = all frames)


def main() -> int:
    if not VIDEO_PATH.exists():
        print(f"Error: {VIDEO_PATH!s} does not exist.", file=sys.stderr)
        return 1

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        print(f"Error: Unable to open {VIDEO_PATH!s}.", file=sys.stderr)
        return 1

    total_reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source video reports {total_reported} frames.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extracted = 0
    saved = 0

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        extracted += 1
        if EVERY_N_FRAMES > 1 and extracted % EVERY_N_FRAMES != 0:
            continue

        filename = OUTPUT_DIR / f"frame_{saved:06d}.jpg"
        cv2.imwrite(str(filename), frame)
        saved += 1

        if FRAME_LIMIT is not None and saved >= FRAME_LIMIT:
            break

    capture.release()
    print(f"Frames read: {extracted} ; frames saved: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
