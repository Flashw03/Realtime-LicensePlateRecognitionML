"""Offline image test: detection + OCR on random test-set images.

Loads models/best.pt, picks random images from the dataset's test split,
runs the full detect → preprocess → OCR → validate chain, saves annotated
images to outputs/visualizations/, and prints recognized plate text.

Usage:
    python scripts/test_images.py                # 8 random test images
    python scripts/test_images.py --num 10 --split valid
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detect import PlateDetector  # noqa: E402
from src.ocr import PlateOCR  # noqa: E402
from src.postprocess import (  # noqa: E402
    RecognitionResult,
    clean_plate_text,
    correct_plate_format,
    validate_plate,
)
from src.preprocess import preprocess_plate  # noqa: E402
from src.utils import get_timestamp, load_config, log_detection, setup_csv_logger  # noqa: E402
from src.visualize import draw_results  # noqa: E402


def find_test_images(data_yaml_path: Path, split: str) -> list:
    """Resolve the image folder for a split via the dataset's data.yaml."""
    cfg = yaml.safe_load(data_yaml_path.read_text())
    root = Path(cfg.get("path", data_yaml_path.parent))
    if not root.is_absolute():
        root = (data_yaml_path.parent / root).resolve()
    rel = cfg.get(split)
    if rel is None:
        return []
    img_dir = root / rel
    return sorted(p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def main():
    parser = argparse.ArgumentParser(description="Test detection + OCR on dataset images")
    parser.add_argument("--num", type=int, default=8, help="Number of images to test (default: 8)")
    parser.add_argument("--split", default="test", choices=["test", "valid", "train"],
                        help="Dataset split to sample from (default: test)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--save-csv", action="store_true",
                        help="Also append results to outputs/logs/detections.csv")
    args = parser.parse_args()

    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    data_yaml_path = PROJECT_ROOT / cfg["training"]["data_yaml"]

    if not data_yaml_path.exists():
        sys.exit(f"❌ Dataset config not found: {data_yaml_path}\n"
                 "   Download the dataset first (kagglehub or python src/train.py --download).")

    images = find_test_images(data_yaml_path, args.split)
    if not images:
        sys.exit(f"❌ No images found for split '{args.split}' under {data_yaml_path}")

    if args.seed is not None:
        random.seed(args.seed)
    samples = random.sample(images, min(args.num, len(images)))

    print(f"🧪 Testing {len(samples)} random '{args.split}' images "
          f"(pool: {len(images)}) with {cfg['detection']['model_path']}\n")

    detector = PlateDetector(cfg)
    ocr = PlateOCR(cfg)

    vis_dir = PROJECT_ROOT / cfg["output"]["visualization_dir"]
    vis_dir.mkdir(parents=True, exist_ok=True)

    csv_fh, csv_writer = None, None
    if args.save_csv:
        csv_fh, csv_writer = setup_csv_logger(str(PROJECT_ROOT / cfg["output"]["log_path"]))

    plates_found = 0
    try:
        for i, img_path in enumerate(samples, 1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"[{i}] {img_path.name} — could not read image, skipping")
                continue

            detections = detector.detect(frame)
            results = []
            for det in detections:
                preprocessed = preprocess_plate(det.plate_image, cfg)
                raw_text, ocr_conf = ocr.recognize(preprocessed)
                cleaned = clean_plate_text(raw_text)
                if cleaned:
                    corrected = correct_plate_format(cleaned)
                    if corrected:
                        cleaned = corrected
                valid = validate_plate(cleaned)

                res = RecognitionResult(
                    plate_text=cleaned or raw_text.upper(),
                    ocr_confidence=ocr_conf,
                    det_confidence=det.det_confidence,
                    bbox=det.bbox,
                    is_valid=valid,
                    frame_no=0,
                    timestamp=get_timestamp(),
                )
                results.append(res)
                if csv_writer is not None and valid:
                    log_detection(csv_writer, res)

            annotated = draw_results(frame, results, fps=0.0)
            out_path = vis_dir / f"test_{args.split}_{i:02d}_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), annotated)

            status = f"saved → {out_path.name}"
            if results:
                plates_found += len(results)
                parts = [f"  🎯 '{r.plate_text}'"
                         f" (ocr={r.ocr_confidence:.2f}, det={r.det_confidence:.2f}"
                         f"{', valid' if r.is_valid else ', INVALID format'})"
                         for r in results]
                print(f"[{i}] {img_path.name} — {len(results)} plate(s):\n" + "\n".join(parts))
                print(f"     {status}")
            else:
                print(f"[{i}] {img_path.name} — no plates detected · {status}")
    finally:
        if csv_fh is not None:
            csv_fh.flush()
            csv_fh.close()

    print(f"\n✅ Done — {plates_found} plate(s) recognized across {len(samples)} images.")
    if csv_writer is not None:
        print("   Results appended to outputs/logs/detections.csv")


if __name__ == "__main__":
    main()
