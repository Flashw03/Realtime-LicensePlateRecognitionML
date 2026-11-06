"""End-to-end ALPR pipeline orchestrator.

Ties together YOLO detection, plate preprocessing, EasyOCR (CPU), regex
validation, temporal deduplication, CSV logging, and annotated real-time
display. Works with pre-trained weights (Level 0) out of the box.
"""

import time
from typing import List, Union

import cv2
import numpy as np

from .detect import PlateDetector
from .ocr import PlateOCR
from .postprocess import (
    PlateConsensus,
    RecognitionResult,
    TemporalDeduplicator,
    clean_plate_text,
    correct_plate_format,
    validate_plate,
)
from .preprocess import preprocess_plate
from .tracker import PlateTracker, csrt_available
from .utils import get_timestamp, load_config, log_detection, setup_csv_logger
from .visualize import draw_results


class ALPRPipeline:
    """Orchestrates the full detect → OCR → validate → log → display loop."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize all components from config.
        Must work with pre-trained weights (Level 0) out of the box.
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        self._configure_devices()

        # Detector errors clearly if best.pt is missing
        try:
            self.detector = PlateDetector(self.config)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"\n❌ {exc}\n"
                "   Download pre-trained weights as described in models/README.md,\n"
                "   save them as models/best.pt, then re-run.\n"
            ) from exc

        print("⏳ Initializing EasyOCR (first run downloads models — may take 2–5 minutes)...")
        self.ocr_engine = PlateOCR(self.config)

        dedup_window = float(self.config["postprocess"].get("dedup_window_seconds", 3.0))
        self.deduplicator = TemporalDeduplicator(dedup_window)
        self.plate_pattern = self.config["postprocess"].get("plate_regex")

        # Multi-frame consensus: tracks plates, emits the best valid reading
        # per plate per emit window (the log carries reliable text, not noise)
        self.consensus = None
        self._last_display_results = []
        self._last_display_detections = []
        if self.config["postprocess"].get("consensus", True):
            self.consensus = PlateConsensus(
                iou_threshold=float(self.config["postprocess"].get("consensus_iou", 0.3)),
                max_age_seconds=float(
                    self.config["postprocess"].get("consensus_max_age_seconds", 2.0)),
                emit_seconds=float(
                    self.config["postprocess"].get("consensus_emit_seconds", 3.0)),
            )
        self.correct_ocr = bool(self.config["postprocess"].get("positional_correction", True))

        # Optional CSRT tracking — degrades gracefully when the OpenCV build
        # has no tracking API (base opencv-python 5.x; use opencv-contrib-python)
        self.tracker = None
        if self.config["video"].get("use_tracker", True):
            if csrt_available():
                self.tracker = PlateTracker()
            else:
                print("⚠️  CSRT tracker unavailable in this OpenCV build — "
                      "running detection-only (install opencv-contrib-python to enable).")

        log_path = self.config["output"]["log_path"]
        self._csv_fh, self._csv_writer = setup_csv_logger(log_path)
        self._plates_logged = 0

    def _configure_devices(self) -> None:
        """Apply the local-VRAM device policy: CPU fallback + EasyOCR off GPU."""
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            if cuda_ok:
                name = torch.cuda.get_device_name(0)
                print(f"🎮 CUDA available: {name}")
        except ImportError:
            cuda_ok = False
            print("⚠️  PyTorch not importable — detection will run on CPU.")

        device = str(self.config["detection"].get("device", "cpu"))
        if device.lower() not in ("cpu", "-1") and not cuda_ok:
            print("⚠️  CUDA not available — falling back to CPU for YOLO detection.")
            self.config["detection"]["device"] = "cpu"

        if self.config["ocr"].get("gpu", False):
            print("⚠️  Overriding ocr.gpu → False: EasyOCR must stay on CPU so YOLO "
                  "owns the GPU (6GB VRAM budget).")
            self.config["ocr"]["gpu"] = False

    def process_frame(self, frame: np.ndarray, frame_no: int) -> List[RecognitionResult]:
        """
        Process a single frame through the full pipeline:
        1. Detect plates (YOLO)
        2. For each detection: preprocess → OCR → clean → correct → validate
        3. With consensus: offer valid readings to plate tracks; emit each
           track's best reading once per emit window (what gets logged)
        4. Return results to log; per-frame valid readings are kept on
           self._last_display_results for live annotation

        Note: bounding box coordinates correspond to the frame as passed in
        (i.e. after any resize applied by run()).
        """
        detections = self.detector.detect(frame)
        if self.tracker is not None:
            self.tracker.update(detections, frame)
        now = time.time()

        # Consensus: OCR each frame's own crop (readings vary with blur),
        # tracks hold the best valid reading per physical plate
        if self.consensus is not None:
            pending = list(self.consensus.prune())
            crops, track_ids = self.consensus.update(detections)
        else:
            crops = [det.plate_image for det in detections]
            track_ids = [None] * len(detections)

        results = []
        display_results = []

        for det, crop, track_id in zip(detections, crops, track_ids):
            preprocessed = preprocess_plate(crop, self.config)
            raw_text, ocr_conf = self.ocr_engine.recognize(preprocessed)
            if not raw_text:
                continue

            cleaned = clean_plate_text(raw_text)
            if self.correct_ocr and cleaned:
                # Positional repair of digit/letter confusions against the
                # Indian plate layout ('HR574Z3919' → 'HR 57 AZ 3919')
                corrected = correct_plate_format(cleaned)
                if corrected:
                    cleaned = corrected
            # Validate: only format-valid readings go further
            if not validate_plate(cleaned, self.plate_pattern):
                continue

            res = RecognitionResult(
                plate_text=cleaned,
                ocr_confidence=ocr_conf,
                det_confidence=det.det_confidence,
                bbox=det.bbox,
                is_valid=True,
                frame_no=frame_no,
                timestamp=get_timestamp(),
            )
            display_results.append(res)

            if self.consensus is not None:
                self.consensus.offer(track_id, ocr_conf, cleaned,
                                     det.bbox, frame_no, det.det_confidence)
            elif not self.deduplicator.is_duplicate(cleaned, now):
                results.append(res)

        if self.consensus is not None:
            pending.extend(self.consensus.poll())
            for best in pending:
                results.append(RecognitionResult(
                    plate_text=best["text"],
                    ocr_confidence=best["conf"],
                    det_confidence=best["det_conf"],
                    bbox=best["bbox"],
                    is_valid=True,
                    frame_no=best["frame_no"],
                    timestamp=get_timestamp(),
                ))

        self._last_display_results = display_results if self.consensus is not None else results
        self._last_display_detections = detections
        return results

    def run(self) -> None:
        """
        Main loop: read frames, skip/process, annotate, display, log.
        Press 'q' in the video window to quit.
        """
        video_cfg = self.config["video"]
        frame_skip = max(1, int(video_cfg.get("frame_skip", 1)))
        resize_w = video_cfg.get("resize_width")
        resize_h = video_cfg.get("resize_height")
        save_frames = self.config["output"].get("save_annotated_frames", False)
        vis_dir = self.config["output"].get("visualization_dir", "outputs/visualizations")

        cap = self._open_source(video_cfg["source"])

        print(f"\n🚀 ALPR pipeline running — source: {video_cfg['source']} · "
              f"frame_skip: {frame_skip} · device: {self.config['detection']['device']}")
        print("Press 'q' in the video window to quit.\n")

        display = True
        window_name = "ALPR — press 'q' to quit"
        frame_no = 0
        processed = 0
        fps_t0 = time.perf_counter()
        fps_ema = 0.0
        last_processed_t = None
        last_results: List[RecognitionResult] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if str(video_cfg["source"]).isdigit():
                        print("\n📹 Webcam stream ended.")
                    else:
                        print("\n🎬 End of video.")
                    break
                frame_no += 1

                if resize_w and resize_h:
                    frame = cv2.resize(frame, (int(resize_w), int(resize_h)))

                tracked_boxes = []
                display_dets = None
                if frame_no % frame_skip == 0:
                    last_results = self.process_frame(frame, frame_no)
                    for res in last_results:
                        log_detection(self._csv_writer, res)
                        self._plates_logged += 1
                    self._csv_fh.flush()  # dashboard can tail the file live
                    processed += 1

                    now_perf = time.perf_counter()
                    if last_processed_t is not None:
                        inst = 1.0 / (now_perf - last_processed_t)
                        fps_ema = inst if fps_ema == 0.0 else 0.8 * fps_ema + 0.2 * inst
                    last_processed_t = now_perf

                    if processed % 30 == 0:
                        elapsed = now_perf - fps_t0
                        if elapsed > 0:
                            print(f"⚡ {processed / elapsed:5.1f} FPS · frame {frame_no} · "
                                  f"{self._plates_logged} plates logged")
                            fps_t0 = now_perf

                    display_dets = self._last_display_detections
                elif self.tracker is not None:
                    # Skipped frame: propagate the last boxes for smooth display
                    tracked_boxes = self.tracker.predict(frame)

                # Live annotation: green box on EVERY detection, text label only
                # on plates whose OCR reading validated this frame
                annotated = draw_results(frame, self._last_display_results,
                                         fps=fps_ema, tracked_boxes=tracked_boxes,
                                         detections=display_dets)

                if display:
                    try:
                        cv2.imshow(window_name, annotated)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            print("\n⏹ Quit requested.")
                            break
                    except cv2.error:
                        # Headless environment — fall back to saving frames
                        display = False
                        print("⚠️  cv2.imshow unavailable (headless?) — "
                              "saving every 30th annotated frame instead.")
                        cv2.destroyAllWindows() if hasattr(cv2, "destroyAllWindows") else None

                if save_frames and frame_no % frame_skip == 0:
                    self._save_frame(annotated, frame_no, vis_dir)
                elif not display and frame_no % 30 == 0:
                    self._save_frame(annotated, frame_no, vis_dir)

        except KeyboardInterrupt:
            print("\n⏹ Interrupted by user.")
        finally:
            cap.release()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            self._csv_fh.flush()
            self._csv_fh.close()
            print(f"\n✅ Done — {self._plates_logged} plates logged to "
                  f"'{self.config['output']['log_path']}'")

    def _open_source(self, source: Union[str, int]):
        """Open a VideoCapture for a file path or webcam index, with clear errors."""
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            cap = cv2.VideoCapture(int(source))
            if not cap.isOpened():
                raise SystemExit(f"❌ Could not open webcam source {source}.")
            return cap

        import os
        if not os.path.exists(source):
            raise SystemExit(
                f"❌ Video file not found: '{source}'\n"
                "   Set video.source in config/config.yaml or pass --source "
                "(use --source 0 for webcam)."
            )
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"❌ Could not open video file: '{source}'")
        return cap

    @staticmethod
    def _save_frame(frame: np.ndarray, frame_no: int, vis_dir: str) -> None:
        """Save an annotated frame to the visualization directory."""
        import os
        os.makedirs(vis_dir, exist_ok=True)
        cv2.imwrite(os.path.join(vis_dir, f"frame_{frame_no:06d}.jpg"), frame)
