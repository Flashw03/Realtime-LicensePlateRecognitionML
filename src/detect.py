"""License plate detection using YOLOv8."""

import os
from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Single license plate detection."""
    bbox: tuple  # (x1, y1, x2, y2) as ints
    det_confidence: float
    plate_image: np.ndarray  # Cropped plate region


class PlateDetector:
    """YOLOv8-based license plate detector."""

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dict (from config.yaml)
        """
        model_path = config["detection"]["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at '{model_path}'. "
                f"See models/README.md for download instructions."
            )
        self.model = YOLO(model_path)
        self.confidence_threshold = config["detection"]["confidence_threshold"]
        self.device = config["detection"]["device"]

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect license plates in a frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            List of Detection objects with cropped plate images
        """
        results = self.model(frame, verbose=False, device=self.device)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Clamp to frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    det_confidence=confidence,
                    plate_image=plate_crop.copy()
                ))

        return detections
