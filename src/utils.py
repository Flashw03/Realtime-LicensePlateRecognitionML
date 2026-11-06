"""Utility functions: config loading, CSV logging, path helpers."""

import csv
import os
from datetime import datetime

import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_csv_logger(log_path: str) -> tuple:
    """
    Initialize CSV logger. Creates file with headers if it doesn't exist.
    Returns (file_handle, csv_writer) — caller must close the file handle.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path)

    fh = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fh)

    if not file_exists:
        writer.writerow([
            "timestamp", "frame_no", "plate_text", "ocr_confidence",
            "det_confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
        ])

    return fh, writer


def log_detection(writer: csv.writer, result) -> None:
    """Write a single RecognitionResult to CSV."""
    writer.writerow([
        result.timestamp,
        result.frame_no,
        result.plate_text,
        f"{result.ocr_confidence:.4f}",
        f"{result.det_confidence:.4f}",
        result.bbox[0], result.bbox[1],
        result.bbox[2], result.bbox[3]
    ])


def get_timestamp() -> str:
    """Return current ISO 8601 timestamp."""
    return datetime.now().isoformat(timespec="milliseconds")
