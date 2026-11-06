"""Post-processing: plate text cleaning, Indian plate validation, temporal dedup."""

import re
import time
from dataclasses import dataclass

# Indian plate format: SS DD LL NNNN (state, district, series, registration)
INDIAN_PLATE_REGEX = re.compile(r'^[A-Z]{2}\s?\d{2}\s?[A-Z]{1,3}\s?\d{1,4}$')


@dataclass
class RecognitionResult:
    """Complete recognition result (detection + OCR)."""
    plate_text: str                    # Recognized text
    ocr_confidence: float              # EasyOCR confidence
    det_confidence: float              # YOLO confidence
    bbox: tuple                        # (x1, y1, x2, y2)
    is_valid: bool                     # Passed regex validation
    frame_no: int                      # Source frame number
    timestamp: str                     # ISO 8601 timestamp


def clean_plate_text(text: str) -> str:
    """Normalize raw OCR output into a canonical plate string.

    Uppercases, converts separator characters (dashes, underscores, dots,
    slashes, commas) to spaces, strips anything that is not A–Z / 0–9 / space,
    and collapses repeated whitespace. Example: "mh-12-ab-1234" → "MH 12 AB 1234".
    """
    upper = text.upper()
    separated = re.sub(r"[\-_./,]", " ", upper)
    alnum = re.sub(r"[^A-Z0-9 ]", "", separated)
    collapsed = re.sub(r"\s+", " ", alnum).strip()
    return collapsed


def validate_plate(text: str, pattern: str = None) -> bool:
    """Return True if the text matches the Indian plate format.

    Args:
        text: Cleaned plate text.
        pattern: Optional regex from config["postprocess"]["plate_regex"].
            Falls back to the compiled INDIAN_PLATE_REGEX when omitted.
    """
    if pattern is None:
        return bool(INDIAN_PLATE_REGEX.fullmatch(text.strip()))
    return bool(re.compile(pattern).fullmatch(text.strip()))


def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union between two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


class PlateConsensus:
    """Accumulates OCR readings per physical plate across frames.

    Detections are matched to plate tracks by IoU. Every frame's own crop is
    OCR'd fresh — readings swing frame to frame with motion blur, so each
    frame gets a chance to be the sharp one. Each track keeps its
    highest-confidence format-valid reading and emits exactly one reading per
    emit window: the log carries the most reliable text instead of every
    noisy variant.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age_seconds: float = 2.0,
                 emit_seconds: float = 3.0):
        """
        Args:
            iou_threshold: Minimum IoU to consider a detection the same plate
                as an existing track.
            max_age_seconds: Tracks unseen for longer than this are pruned
                (emitting their pending best reading first).
            emit_seconds: Emit each track's best reading once per this many
                seconds.
        """
        self.iou_threshold = iou_threshold
        self.max_age_seconds = max_age_seconds
        self.emit_seconds = emit_seconds
        self._tracks = []  # dicts: bbox, last_seen, best, first_valid

    def update(self, detections) -> tuple:
        """Match detections to tracks; return the crops to OCR this frame.

        Args:
            detections: Detection-like objects exposing `.bbox` and
                `.plate_image`.

        Returns:
            (crops, track_ids) aligned with `detections` — the CURRENT crop of
            each detection plus the track index it matched (new tracks are
            appended).
        """
        now = time.time()
        crops, ids = [], []
        unmatched = list(range(len(self._tracks)))

        for det in detections:
            best_idx, best_iou = None, 0.0
            for idx in unmatched:
                iou = _iou(self._tracks[idx]["bbox"], det.bbox)
                if iou > best_iou:
                    best_idx, best_iou = idx, iou

            if best_idx is not None and best_iou >= self.iou_threshold:
                idx = best_idx
                unmatched.remove(idx)
            else:
                self._tracks.append({"bbox": det.bbox, "last_seen": now,
                                     "best": None, "first_valid": None})
                idx = len(self._tracks) - 1

            self._tracks[idx]["bbox"] = det.bbox
            self._tracks[idx]["last_seen"] = now
            crops.append(det.plate_image)
            ids.append(idx)

        return crops, ids

    def offer(self, track_id: int, ocr_conf: float, text: str,
              bbox: tuple, frame_no: int, det_conf: float) -> None:
        """Record a format-valid reading; kept if it beats the track's best."""
        track = self._tracks[track_id]
        now = time.time()
        if track["best"] is None:
            track["first_valid"] = now
        if track["best"] is None or ocr_conf > track["best"]["conf"]:
            track["best"] = {"conf": ocr_conf, "text": text, "bbox": bbox,
                             "frame_no": frame_no, "det_conf": det_conf}

    def poll(self) -> list:
        """Emit each track's best reading once per emit window."""
        now = time.time()
        emitted = []
        for track in self._tracks:
            best, first = track["best"], track["first_valid"]
            if best is None or first is None:
                continue
            if now - first >= self.emit_seconds:
                emitted.append(best)
                track["best"] = None
                track["first_valid"] = None
        return emitted

    def prune(self) -> list:
        """Drop stale tracks, emitting their pending best reading first."""
        now = time.time()
        alive, emitted = [], []
        for track in self._tracks:
            if now - track["last_seen"] <= self.max_age_seconds:
                alive.append(track)
            elif track["best"] is not None:
                emitted.append(track["best"])
        self._tracks = alive
        return emitted

    def reset(self) -> None:
        """Forget all plate tracks."""
        self._tracks.clear()


# Common OCR class confusions, applied positionally by the plate layout:
# letter slots may hold a misread digit, digit slots a misread letter.
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
_LETTER_TO_DIGIT = {v: k for k, v in _DIGIT_TO_LETTER.items()}

# Segmentations (series letters, registration digits) for the Indian plate
# layout SS DD LL NNNN, most common private-plate shape first.
_SEGMENTATIONS = [(2, 4), (1, 4), (3, 4), (2, 3), (1, 3), (3, 3),
                  (2, 2), (1, 2), (3, 2), (2, 1), (1, 1), (3, 1)]


def correct_plate_format(text: str, max_corrections: int = 1) -> "str | None":
    """Repair near-miss OCR readings using the Indian plate layout.

    Splits the compact reading into SS DD LL NNNN segments and fixes
    digit/letter class confusions per position (e.g. 'HR574Z3919' →
    'HR 57 AZ 3919', where '4' occupies a letter slot). Also normalizes
    spacing on readings that are already correct.

    Args:
        text: Cleaned (uppercase, space-separated) OCR text.
        max_corrections: Maximum character-class substitutions allowed.
            Conservative by design: a wrongly "corrected" plate is worse
            than an unlogged one.

    Returns:
        The corrected, space-separated plate if it becomes format-valid,
        otherwise None (caller keeps the original text).
    """
    compact = text.replace(" ", "")
    total = len(compact)
    if not 6 <= total <= 11:
        return None

    for ll, nn in _SEGMENTATIONS:
        if ll + nn != total - 4:
            continue

        fixes = 0

        def to_letter(ch: str) -> str:
            nonlocal fixes
            if ch.isdigit():
                fixes += 1
                return _DIGIT_TO_LETTER.get(ch, ch)
            return ch

        def to_digit(ch: str) -> str:
            nonlocal fixes
            if ch.isalpha():
                fixes += 1
                return _LETTER_TO_DIGIT.get(ch, ch)
            return ch

        ss = "".join(to_letter(c) for c in compact[0:2])
        dd = "".join(to_digit(c) for c in compact[2:4])
        series = "".join(to_letter(c) for c in compact[4:4 + ll])
        reg = "".join(to_digit(c) for c in compact[4 + ll:])

        if fixes > max_corrections:
            continue
        if not (ss.isalpha() and dd.isdigit() and series.isalpha() and reg.isdigit()):
            continue

        candidate = f"{ss} {dd} {series} {reg}"
        if validate_plate(candidate):
            return candidate

    return None


class TemporalDeduplicator:
    """Suppresses repeated detections of the same plate within a time window."""

    def __init__(self, window_seconds: float = 3.0):
        """
        Args:
            window_seconds: Same plate text within this many seconds is a duplicate.
        """
        self.window_seconds = window_seconds
        self._last_seen = {}  # plate_text -> timestamp of last accepted detection

    def is_duplicate(self, plate_text: str, timestamp: float) -> bool:
        """
        Check (and record) a detection.

        Args:
            plate_text: Cleaned plate text.
            timestamp: Current time in seconds (e.g. time.time()).

        Returns:
            True if this plate was already seen within the window.
        """
        last = self._last_seen.get(plate_text)
        if last is not None and (timestamp - last) < self.window_seconds:
            return True
        self._last_seen[plate_text] = timestamp
        return False

    def reset(self) -> None:
        """Forget all recently seen plates."""
        self._last_seen.clear()
