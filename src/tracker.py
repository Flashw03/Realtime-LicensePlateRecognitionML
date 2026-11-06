"""Simple multi-object tracker for license plates using OpenCV CSRT.

Bridges the gap between YOLO detection frames: on frames that are skipped
by the pipeline, `predict()` propagates known boxes forward so annotation
stays smooth. Only positions are tracked — no re-identification.
"""

from typing import List, Tuple, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:  # Avoid importing ultralytics just to type-hint
    from .detect import Detection


def csrt_available() -> bool:
    """True if a CSRT tracker can be created in this OpenCV build.

    The base opencv-python 5.x wheels dropped the legacy tracking API —
    install opencv-contrib-python to enable tracking.
    """
    try:
        _create_csrt_tracker()
        return True
    except RuntimeError:
        return False


def _create_csrt_tracker():
    """Create a CSRT tracker with a fallback to the contrib/legacy variant."""
    try:
        return cv2.TrackerCSRT_create()
    except (AttributeError, cv2.error):
        pass
    try:
        return cv2.legacy.TrackerCSRT_create()
    except AttributeError as exc:
        raise RuntimeError(
            "CSRT tracker is unavailable in this OpenCV build "
            "(neither cv2.TrackerCSRT_create nor cv2.legacy.TrackerCSRT_create). "
            "Install opencv-python >= 4.5.2."
        ) from exc


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


class PlateTracker:
    """Tracks license plate bounding boxes between detection frames."""

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 5):
        """
        Args:
            iou_threshold: Minimum IoU between a detection and a track to match.
            max_misses: Consecutive detection frames without a match before a
                track is dropped (plate left the frame).
        """
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._tracks = []  # each: dict(id, tracker, bbox, misses)
        self._next_id = 1

    def update(self, detections: List["Detection"], frame: np.ndarray) -> List["Detection"]:
        """
        Sync tracker state with fresh YOLO detections.

        Advances every track on `frame`, greedily matches detections to tracks
        by IoU (re-initializing matched trackers with the YOLO box), starts new
        tracks for unmatched detections, and ages out stale tracks.

        Args:
            detections: Detections from the current frame (authoritative boxes).
            frame: Current BGR frame.

        Returns:
            The same list of detections, passed through unchanged.
        """
        self._advance(frame)

        unmatched_tracks = list(range(len(self._tracks)))
        for det in detections:
            best_idx, best_iou = None, 0.0
            for idx in unmatched_tracks:
                iou = _iou(self._tracks[idx]["bbox"], det.bbox)
                if iou > best_iou:
                    best_idx, best_iou = idx, iou

            if best_idx is not None and best_iou >= self.iou_threshold:
                track = self._tracks[best_idx]
                unmatched_tracks.remove(best_idx)
                track["tracker"] = _create_csrt_tracker()
                track["tracker"].init(frame, self._to_xywh(det.bbox))
                track["bbox"] = det.bbox
                track["misses"] = 0
            else:
                tracker = _create_csrt_tracker()
                tracker.init(frame, self._to_xywh(det.bbox))
                self._tracks.append({
                    "id": self._next_id,
                    "tracker": tracker,
                    "bbox": det.bbox,
                    "misses": 0,
                })
                self._next_id += 1

        # Age out tracks that no longer match any detection
        for idx in unmatched_tracks:
            self._tracks[idx]["misses"] += 1
        self._tracks = [t for t in self._tracks if t["misses"] <= self.max_misses]

        return detections

    def predict(self, frame: np.ndarray) -> List[Tuple[int, tuple]]:
        """
        Advance all tracks on a skipped frame (no new detections).

        Args:
            frame: Current BGR frame.

        Returns:
            List of (track_id, (x1, y1, x2, y2)) for tracks still alive.
        """
        self._advance(frame)
        return [(t["id"], t["bbox"]) for t in self._tracks]

    def _advance(self, frame: np.ndarray) -> None:
        """Move every tracker one frame forward; drop ones that fail."""
        alive = []
        for track in self._tracks:
            ok, box = track["tracker"].update(frame)
            if ok:
                track["bbox"] = self._to_xyxy(box)
                alive.append(track)
        self._tracks = alive

    @staticmethod
    def _to_xywh(bbox: tuple) -> tuple:
        """Convert (x1, y1, x2, y2) to OpenCV tracker (x, y, w, h)."""
        x1, y1, x2, y2 = bbox
        return (x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def _to_xyxy(box) -> tuple:
        """Convert OpenCV tracker (x, y, w, h) to (x1, y1, x2, y2) ints."""
        x, y, w, h = box
        return (int(x), int(y), int(x + w), int(y + h))
