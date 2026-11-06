"""Video overlay styling for the ALPR pipeline.

Draws clean, minimal detection overlays: green bounding boxes, dark
semi-transparent label bars with plate text + confidence, an idle
message when nothing is detected, and an FPS counter.

This module is a standalone visual utility — the pipeline keeps its own
simple annotation and may adopt this styling incrementally.
"""

from typing import List

import cv2
import numpy as np

# Palette (BGR)
BOX_COLOR = (100, 220, 0)        # green border
LABEL_BG = (40, 40, 40)          # dark bar
TEXT_COLOR = (255, 255, 255)     # white
IDLE_COLOR = (170, 170, 170)     # gray
FPS_COLOR = (150, 150, 150)      # gray

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
BOX_THICKNESS = 2
LABEL_ALPHA = 0.65               # transparency of the label bar
LABEL_PADDING = 6


def _blend_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                color: tuple, alpha: float) -> None:
    """Draw an alpha-blended solid rectangle in place."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)


def draw_results(
    frame: np.ndarray,
    results: list,  # List of RecognitionResult
    fps: float = 0.0,
    tracked_boxes: list = None,  # [(track_id, (x1, y1, x2, y2))] on skipped frames
    detections: list = None,  # List of Detection — every plate the detector found
) -> np.ndarray:
    """
    Draw styled detection overlays on frame.
    Returns a new frame (does not modify original).

    Every entry in `detections` gets a green box (the detector's raw output);
    entries in `results` additionally get the text label (OCR validated).
    """
    annotated = frame.copy()

    if not results and not tracked_boxes and not detections:
        cv2.putText(annotated, "No plates detected", (12, 28),
                    FONT, FONT_SCALE, IDLE_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    # Tracker-propagated boxes on skipped frames: dim outline, no label
    for _tid, bbox in (tracked_boxes or []):
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 180, 60), 1, cv2.LINE_AA)

    # All current detections: green box (recognized plates get a label below)
    for det in (detections or []):
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR,
                      BOX_THICKNESS, cv2.LINE_AA)

    for res in results:
        x1, y1, x2, y2 = (int(v) for v in res.bbox)
        if detections:
            # Box already drawn from the detection above — label only
            pass
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR,
                          BOX_THICKNESS, cv2.LINE_AA)

        # Label: plate text + detection confidence, above the box
        label = f"{res.plate_text}  {res.det_confidence * 100:.0f}%"
        (tw, th), baseline = cv2.getTextSize(
            label, FONT, FONT_SCALE, FONT_THICKNESS)

        bar_h = th + 2 * LABEL_PADDING
        bar_y2 = y1 - 4
        bar_y1 = bar_y2 - bar_h
        if bar_y1 < 0:  # box touches the top — put the label inside
            bar_y1, bar_y2 = y1 + 4, y1 + 4 + bar_h
        _blend_rect(annotated, x1, bar_y1, x1 + tw + 2 * LABEL_PADDING,
                    bar_y2, LABEL_BG, LABEL_ALPHA)
        cv2.putText(annotated, label, (x1 + LABEL_PADDING,
                                       bar_y2 - LABEL_PADDING - baseline),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    if fps > 0:
        cv2.putText(annotated, f"FPS: {fps:.1f}", (12, annotated.shape[0] - 12),
                    FONT, FONT_SCALE, FPS_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated
