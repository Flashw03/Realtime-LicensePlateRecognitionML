"""OCR pipeline using EasyOCR."""

from typing import Tuple

import numpy as np
import easyocr


class PlateOCR:
    """EasyOCR-based license plate text recognizer."""

    def __init__(self, config: dict):
        """
        Initialize EasyOCR reader.
        IMPORTANT: gpu=False to avoid CUDA OOM (YOLO uses the GPU).
        """
        self.reader = easyocr.Reader(
            config["ocr"]["languages"],
            gpu=config["ocr"]["gpu"]  # Must be False for RTX 4050
        )
        self.confidence_threshold = config["ocr"]["confidence_threshold"]

    def recognize(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Extract text from preprocessed plate image.

        Args:
            plate_image: Preprocessed grayscale plate image

        Returns:
            (plate_text, confidence) tuple
        """
        results = self.reader.readtext(plate_image)

        if not results:
            return ("", 0.0)

        texts = []
        confidences = []
        for (bbox, text, prob) in results:
            if prob >= self.confidence_threshold:
                texts.append(text.upper().strip())
                confidences.append(prob)

        if not texts:
            return ("", 0.0)

        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences)

        return (combined_text, avg_confidence)
