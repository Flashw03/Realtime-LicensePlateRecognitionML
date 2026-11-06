"""Unit & integration tests for the ALPR pipeline.

All tests pass WITHOUT a GPU, trained model, or EasyOCR — they use
synthetic data and only exercise the light-weight modules.
"""

import csv
import time

import pytest

from src.postprocess import (
    RecognitionResult,
    TemporalDeduplicator,
    clean_plate_text,
    validate_plate,
)
from src.utils import load_config, log_detection, setup_csv_logger

EXPECTED_HEADERS = [
    "timestamp", "frame_no", "plate_text", "ocr_confidence",
    "det_confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
]


class TestConfig:
    def test_config_loading(self):
        config = load_config("config/config.yaml")
        for key in ("video", "detection", "ocr", "preprocess",
                    "postprocess", "training", "output", "dashboard"):
            assert key in config, f"missing config section: {key}"
        assert config["detection"]["model_path"] == "models/best.pt"
        assert config["ocr"]["gpu"] is False, "EasyOCR must default to CPU"


class TestPreprocess:
    def test_preprocess_output_shape(self):
        cv2 = pytest.importorskip("cv2")
        import numpy as np
        from src.preprocess import preprocess_plate

        config = {"preprocess": {
            "grayscale": True,
            "bilateral_filter": {"d": 11, "sigma_color": 17, "sigma_space": 17},
            "adaptive_threshold": True,
        }}
        synthetic = np.random.randint(0, 255, (48, 160, 3), dtype=np.uint8)
        out = preprocess_plate(synthetic, config)

        assert out.ndim == 2, "preprocessed image must be grayscale"
        assert out.size > 0, "preprocessed image must be non-empty"
        assert set(np.unique(out)).issubset({0, 255}), "adaptive threshold must binarize"


class TestValidation:
    @pytest.mark.parametrize("plate", [
        "MH 12 AB 1234",
        "DL 01 C 5678",
        "KA 05 BH 9012",
    ])
    def test_plate_validation_valid(self, plate):
        assert validate_plate(plate) is True

    @pytest.mark.parametrize("plate", ["HELLO", "123", "", "MH-12-AB"])
    def test_plate_validation_invalid(self, plate):
        assert validate_plate(plate) is False

    def test_clean_plate_text(self):
        assert clean_plate_text("MH-12-AB-1234") == "MH 12 AB 1234"
        assert clean_plate_text("mh 12 ab 1234") == "MH 12 AB 1234"
        assert clean_plate_text("  MH   12  AB  1234 ") == "MH 12 AB 1234"
        assert clean_plate_text("MH.12.AB.1234") == "MH 12 AB 1234"


class TestTemporalDedup:
    def test_temporal_dedup(self):
        dedup = TemporalDeduplicator(window_seconds=1.0)
        t0 = time.time()

        assert dedup.is_duplicate("MH 12 AB 1234", t0) is False, "first sighting"
        assert dedup.is_duplicate("MH 12 AB 1234", t0 + 0.5) is True, "within window"
        assert dedup.is_duplicate("DL 01 C 5678", t0 + 0.5) is False, "different plate"

        # Push the last-seen time outside the window → no longer a duplicate
        dedup._last_seen["MH 12 AB 1234"] = t0 - 10.0
        assert dedup.is_duplicate("MH 12 AB 1234", t0) is False, "outside window"


class TestPositionalCorrection:
    def test_fixes_digit_in_letter_slot(self):
        from src.postprocess import correct_plate_format
        # '4' in the series-letter slot → 'A' (1 correction)
        assert correct_plate_format("HR574Z3919") == "HR 57 AZ 3919"

    def test_normalizes_spacing_when_already_valid(self):
        from src.postprocess import correct_plate_format
        assert correct_plate_format("MH12AB1234") == "MH 12 AB 1234"
        assert correct_plate_format("KA 05 BH 9012") == "KA 05 BH 9012"

    def test_garbage_is_not_corrected(self):
        from src.postprocess import correct_plate_format
        # 'HELLO12' would need 2+ class fixes in the DD slot alone
        assert correct_plate_format("HELLO12") is None

    def test_wrong_length_rejected(self):
        from src.postprocess import correct_plate_format
        assert correct_plate_format("123") is None
        assert correct_plate_format("MH12AB12345678") is None

    def test_needs_two_fixes_rejected(self):
        from src.postprocess import correct_plate_format
        # 'DL1C5678': no segmentation fits with ≤1 correction — conservative None
        assert correct_plate_format("DL1C5678") is None

    def test_fixes_letter_in_digit_slot(self):
        from src.postprocess import correct_plate_format
        # 'O' in the district-digit slot → '0' (1 correction)
        assert correct_plate_format("MHO5AB123") == "MH 05 AB 123"


class TestPlateConsensus:
    """Unit tests for the multi-frame reading consensus (pure logic, no OCR)."""

    @staticmethod
    def _det(bbox, h=40, w=120):
        import numpy as np
        from types import SimpleNamespace
        return SimpleNamespace(bbox=bbox,
                               plate_image=np.zeros((h, w, 3), dtype=np.uint8))

    def test_update_returns_current_crops_and_track_ids(self):
        from src.postprocess import PlateConsensus
        cons = PlateConsensus()
        crops, ids = cons.update([self._det((0, 0, 100, 40))])
        assert crops[0].shape == (40, 120, 3) and ids == [0]
        # Same plate, moved box → matched track, but the CURRENT crop is returned
        crops, ids = cons.update([self._det((5, 2, 105, 42), h=42, w=100)])
        assert crops[0].shape == (42, 100, 3) and ids == [0]

    def test_far_apart_detections_are_separate_tracks(self):
        from src.postprocess import PlateConsensus
        cons = PlateConsensus()
        _, ids = cons.update([self._det((0, 0, 100, 40)),
                              self._det((500, 400, 650, 450))])
        assert len(set(ids)) == 2

    def test_emits_best_reading_per_window(self):
        from src.postprocess import PlateConsensus
        cons = PlateConsensus(emit_seconds=3.0)

        _, ids = cons.update([self._det((0, 0, 100, 40))])
        cons.offer(ids[0], 0.5, "HR 57 AZ 3919", (0, 0, 100, 40), 10, 0.8)

        # A sharper frame produces a higher-confidence reading of the same plate
        cons.update([self._det((2, 1, 102, 41))])
        cons.offer(0, 0.9, "HR 51 AZ 3919", (2, 1, 102, 41), 14, 0.8)

        assert cons.poll() == [], "window has not elapsed yet"

        cons._tracks[0]["first_valid"] -= 5.0  # simulate the window elapsing
        emitted = cons.poll()
        assert len(emitted) == 1
        assert emitted[0]["text"] == "HR 51 AZ 3919", "best reading must win"
        assert emitted[0]["conf"] == 0.9
        assert cons.poll() == [], "best is consumed after emission"

    def test_new_window_collects_fresh_readings(self):
        from src.postprocess import PlateConsensus
        cons = PlateConsensus(emit_seconds=3.0)
        _, ids = cons.update([self._det((0, 0, 100, 40))])
        cons.offer(ids[0], 0.9, "HR 51 AZ 3919", (0, 0, 100, 40), 10, 0.8)
        cons._tracks[0]["first_valid"] -= 5.0
        cons.poll()  # first emission

        cons.offer(0, 0.6, "HR 51 AZ 3919", (1, 1, 101, 41), 40, 0.8)
        cons._tracks[0]["first_valid"] -= 5.0
        emitted = cons.poll()
        assert len(emitted) == 1 and emitted[0]["conf"] == 0.6

    def test_prune_emits_pending_reading(self):
        from src.postprocess import PlateConsensus
        cons = PlateConsensus(max_age_seconds=2.0)
        _, ids = cons.update([self._det((0, 0, 100, 40))])
        cons.offer(ids[0], 0.7, "MH 12 AB 1234", (0, 0, 100, 40), 5, 0.9)
        cons._tracks[0]["last_seen"] -= 10.0  # plate left the frame
        emitted = cons.prune()
        assert len(emitted) == 1 and emitted[0]["text"] == "MH 12 AB 1234"
        assert len(cons._tracks) == 0


class TestCsvLogger:
    def test_csv_logger(self, tmp_path):
        log_path = str(tmp_path / "logs" / "detections.csv")

        fh, writer = setup_csv_logger(log_path)
        try:
            result = RecognitionResult(
                plate_text="MH 12 AB 1234",
                ocr_confidence=0.91,
                det_confidence=0.87,
                bbox=(10, 20, 30, 40),
                is_valid=True,
                frame_no=42,
                timestamp="2026-10-27T14:30:05.123",
            )
            log_detection(writer, result)
        finally:
            fh.close()

        with open(log_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert len(rows) == 2, "header + one data row"
        assert rows[0] == EXPECTED_HEADERS
        assert rows[1][2] == "MH 12 AB 1234"
        assert rows[1][1] == "42"
