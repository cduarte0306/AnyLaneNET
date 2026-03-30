from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGES_DIR = Path("data/raw/images")
MASKS_DIR = Path("data/raw/masks")


def extract_lane_mask(frame, ksize=7, sigma=3, contour_area=5000):
    """Return the binary sidewalk/lane mask (0 or 255) for *frame*."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    k = max(1, ksize) * 2 + 1
    blur = cv2.GaussianBlur(hsv, (k, k), max(sigma, 1))

    # Sidewalk: low hue, low saturation, high value
    lower_sidewalk = np.array([5, 0, 150])
    upper_sidewalk = np.array([30, 50, 255])
    sidewalk_mask = cv2.inRange(blur, lower_sidewalk, upper_sidewalk)

    # Mask the grayscale frame to the sidewalk region and binarise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (k, k), max(sigma, 1))
    gray_masked = cv2.bitwise_and(gray_blur, gray_blur, mask=sidewalk_mask)
    gray_masked[gray_masked > 0] = 255

    # Close small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    gray_masked = cv2.morphologyEx(gray_masked, cv2.MORPH_CLOSE, kernel)

    # Remove remaining small black blobs
    contours, _ = cv2.findContours(
        cv2.bitwise_not(gray_masked), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        if cv2.contourArea(cnt) < contour_area:
            cv2.drawContours(gray_masked, [cnt], -1, 255, -1)

    return gray_masked


def generate_dataset(video_path: str, *, preview: bool = False) -> None:
    """Read *video_path* frame-by-frame, extract lane masks, and save pairs
    to ``data/raw/images/`` and ``data/raw/masks/``."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Could not open video: %s", video_path)
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Processing %d frames from %s", total, video_path)

    saved = 0
    idx = 0
    
    # Save every other image of the video 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_ = frame.copy()
        mask = extract_lane_mask(frame_)
        
        # Mask the top quarter
        frame_[: frame_.shape[0] * 2 // 6] = 0
        mask[: mask.shape[0] * 2 // 6] = 0

        # Skip frames where the mask is entirely empty (nothing detected)
        if mask.max() == 0:
            idx += 1
            continue

        name = f"frame_{idx:06d}.png"
        cv2.imwrite(str(IMAGES_DIR / name), frame_)
        cv2.imwrite(str(MASKS_DIR / name), mask)
        saved += 1

        if preview:
            overlay = frame.copy()
            overlay[mask > 0] = (0, 255, 0)
            blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            cv2.imshow("Preview (q to quit)", blended)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        idx += 1

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    log.info("Saved %d image/mask pairs to %s and %s", saved, IMAGES_DIR, MASKS_DIR)


def main():
    
    parser = argparse.ArgumentParser(description="Generate lane-detection training data from video")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--preview", action="store_true", help="Show overlay preview while generating")
    args = parser.parse_args()

    generate_dataset(args.video, preview=args.preview)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()
