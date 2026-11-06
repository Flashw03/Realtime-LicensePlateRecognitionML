"""Image preprocessing utilities for license plate OCR.

Chain: grayscale → bilateral filter → adaptive threshold → optional resize.
All parameters come from config["preprocess"] (see config/config.yaml).
"""

import cv2
import numpy as np


def preprocess_plate(image: np.ndarray, config: dict) -> np.ndarray:
    """Preprocess cropped plate image for OCR.

    Args:
        image: Cropped BGR (or grayscale) plate region from the detector.
        config: Full config dict; parameters are read from config["preprocess"].

    Returns:
        Preprocessed image (grayscale, filtered, binarized).
    """
    cfg = config.get("preprocess", {})

    # 1. Grayscale conversion
    if cfg.get("grayscale", True) and image.ndim == 3:
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        processed = image

    # 2. Bilateral filter — denoise while keeping plate character edges sharp
    bf = cfg.get("bilateral_filter", {})
    if bf:
        processed = cv2.bilateralFilter(
            processed,
            d=int(bf.get("d", 11)),
            sigmaColor=float(bf.get("sigma_color", 17)),
            sigmaSpace=float(bf.get("sigma_space", 17)),
        )

    # 3. Adaptive threshold — handle uneven illumination on plates
    if cfg.get("adaptive_threshold", True):
        processed = cv2.adaptiveThreshold(
            processed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=int(cfg.get("adaptive_block_size", 11)),
            C=float(cfg.get("adaptive_C", 2)),
        )

    # 4. Optional resize to a consistent height (keeps aspect ratio)
    target_h = cfg.get("resize_height")
    if target_h and processed.shape[0] > 0:
        target_h = int(target_h)
        scale = target_h / processed.shape[0]
        if scale != 1.0:
            new_w = max(1, int(round(processed.shape[1] * scale)))
            processed = cv2.resize(processed, (new_w, target_h),
                                   interpolation=cv2.INTER_CUBIC)

    return processed
