"""OCR A/B experiment: EasyOCR vs Tesseract on identical crops.

Runs the Level 1 detector over the 150-image benchmark set (seed 7) and the
video debug frames, then feeds the SAME crops to three OCR paths:

  1. easyocr      — current pipeline path (preprocess_plate + PlateOCR)
  2. tesseract    — upscale to h100 + --psm 7 + A-Z0-9 whitelist
  3. tesseract-otsu — same, but with Otsu binarization first (article's approach)

Every reading goes through clean_plate_text + correct_plate_format, and we
report valid-rate (the metric that actually gates logging) plus timing.

Usage:
    python scripts/ocr_ab_test.py [--num 150] [--seed 7]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytesseract  # noqa: E402

from src.detect import PlateDetector  # noqa: E402
from src.ocr import PlateOCR  # noqa: E402
from src.postprocess import clean_plate_text, correct_plate_format, validate_plate  # noqa: E402
from src.preprocess import preprocess_plate  # noqa: E402
from src.utils import load_config  # noqa: E402
from scripts.test_images import find_test_images  # noqa: E402

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def upscale(image, height=100):
    scale = height / image.shape[0]
    if scale == 1.0:
        return image
    return cv2.resize(image, (max(1, int(image.shape[1] * scale)), height),
                      interpolation=cv2.INTER_CUBIC)


def tesseract_read(crop, binarize=False):
    """Tesseract on an upscaled (optionally Otsu-binarized) crop. -> (text, conf)"""
    img = upscale(crop, 100)
    if binarize:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    data = pytesseract.image_to_data(
        img, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        if text and float(conf) >= 0:
            words.append(text)
            confs.append(float(conf))
    if not words:
        return "", 0.0
    return "".join(words), sum(confs) / len(confs) / 100.0


def evaluate(crops, label, engine_fn, results):
    t0 = time.perf_counter()
    valid, nonempty = 0, 0
    readings = []
    for crop in crops:
        raw, conf = engine_fn(crop)
        cleaned = clean_plate_text(raw)
        corrected = correct_plate_format(cleaned) or cleaned
        is_valid = validate_plate(corrected)
        valid += int(is_valid)
        nonempty += int(bool(raw))
        readings.append((raw, conf, corrected if is_valid else None))
    dt = time.perf_counter() - t0
    results[label] = {
        "nonempty": nonempty, "valid": valid,
        "ms_per_crop": dt / max(1, len(crops)) * 1000,
        "readings": readings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-frames", type=int, nargs="*", default=[60, 128, 132, 136],
                        help="Video frame indices for the head-to-head printout")
    args = parser.parse_args()

    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    detector = PlateDetector(cfg)
    easyocr_engine = PlateOCR(cfg)

    # ── Collect crops: dataset benchmark set + video frames ─────────────
    data_yaml = PROJECT_ROOT / cfg["training"]["data_yaml"]
    images = find_test_images(data_yaml, "test")
    if args.seed is not None:
        import random
        random.seed(args.seed)
    samples = random.sample(images, min(args.num, len(images)))

    crops = []
    print(f"Detecting plates on {len(samples)} images (Level 1 model)...")
    for p in samples:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        crops.extend(d.plate_image for d in detector.detect(frame))
    for fno in args.video_frames:
        cap = cv2.VideoCapture(str(PROJECT_ROOT / cfg["video"]["source"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        cap.release()
        if ok:
            crops.extend(d.plate_image for d in detector.detect(frame))
    print(f"Collected {len(crops)} crops (hard slices included)\n")

    # ── Run the three engines on identical crops ─────────────────────────
    results = {}
    evaluate(crops, "easyocr (current)", lambda c: easyocr_engine.recognize(
        preprocess_plate(c, cfg)), results)
    evaluate(crops, "tesseract (raw h100)", lambda c: tesseract_read(c), results)
    evaluate(crops, "tesseract (otsu h100)", lambda c: tesseract_read(c, binarize=True), results)

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n{'engine':24s} {'nonempty':>9s} {'VALID':>6s} {'valid %':>8s} {'ms/crop':>8s}")
    print("-" * 60)
    for label, r in results.items():
        print(f"{label:24s} {r['nonempty']:>9d} {r['valid']:>6d} "
              f"{100 * r['valid'] / len(crops):>7.1f}% {r['ms_per_crop']:>8.1f}")

    print(f"\n(crops: {len(crops)} · correction: positional, max 1 fix · "
          f"valid = passes Indian plate regex after correction)")

    # ── Head-to-head on the video frames the user has seen ───────────────
    print("\n── Video frame head-to-head ──")
    cap = cv2.VideoCapture(str(PROJECT_ROOT / cfg["video"]["source"]))
    for fno in args.video_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            continue
        dets = detector.detect(frame)
        if not dets:
            print(f"frame {fno}: no detections")
            continue
        big = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        e_raw, e_conf = easyocr_engine.recognize(preprocess_plate(big.plate_image, cfg))
        t_raw, t_conf = tesseract_read(big.plate_image)
        t_otsu_raw, _ = tesseract_read(big.plate_image, binarize=True)
        print(f"frame {fno} (det {big.det_confidence:.2f}): "
              f"easyocr={e_raw!r}@{e_conf:.2f}  tess={t_raw!r}@{t_conf:.2f}  tess-otsu={t_otsu_raw!r}")
    cap.release()


if __name__ == "__main__":
    main()
