"""Performance benchmark charts for the ALPR system.

Generates four charts into outputs/visualizations/ (300 DPI, white background):

1. fps_vs_threshold.png     — FPS across detection confidence thresholds (needs
                              models/best.pt and a video source)
2. detection_heatmap.png    — detection density over time from detections.csv
3. ocr_conf_by_state.png    — OCR confidence box plot grouped by plate state code
4. latency_breakdown.png    — stacked YOLO / preprocess+OCR latency bar (needs best.pt)

Each section skips itself with a clear reason when its inputs are missing,
so the script can run at any point in the project lifecycle.

Usage:
    python scripts/benchmark.py [--video data/sample.mp4] [--frames 300]
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

C_PRIMARY = "#4a90d9"
C_ACCENT = "#2ecc71"
C_ALERT = "#e74c3c"

VIS_DIR = PROJECT_ROOT / "outputs" / "visualizations"
CSV_PATH = PROJECT_ROOT / "outputs" / "logs" / "detections.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})


def _save(fig, name):
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    out = VIS_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ saved {out.relative_to(PROJECT_ROOT)}")


def fps_vs_threshold(video: str, frames: int) -> None:
    """Chart 1: FPS at detection thresholds 0.1–0.9."""
    if not MODEL_PATH.exists():
        print("   ⏭️  skipped (models/best.pt missing — add weights first)")
        return
    import cv2
    from src.utils import load_config

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"   ⏭️  skipped (video '{video}' could not be opened)")
        return
    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    device = cfg["detection"].get("device", "cpu")

    grabbed = []
    while len(grabbed) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        grabbed.append(frame)
    cap.release()
    if not grabbed:
        print("   ⏭️  skipped (no frames could be read)")
        return

    thresholds = np.arange(0.1, 1.0, 0.1)
    fps_values = []
    for t in thresholds:
        t0 = time.perf_counter()
        for frame in grabbed:
            model.predict(frame, conf=float(t), device=device, verbose=False)
        elapsed = time.perf_counter() - t0
        fps_values.append(len(grabbed) / elapsed)
        print(f"   threshold {t:.1f}: {fps_values[-1]:6.1f} FPS")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, fps_values, marker="o", color=C_PRIMARY, linewidth=2)
    ax.axhline(15, color=C_ALERT, linestyle="--", linewidth=1.2, label="Real-time target (15 FPS)")
    ax.set_title("FPS vs Detection Confidence Threshold")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Frames per second")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fps_vs_threshold.png")


def detection_heatmap(df: pd.DataFrame) -> None:
    """Chart 2: detection density per minute across the session."""
    if df.empty:
        print("   ⏭️  skipped (no detection data)")
        return
    per_min = df.set_index("timestamp").resample("1min").size()
    matrix = per_min.values.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(12, 2.8))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    step = max(1, len(per_min) // 12)
    ax.set_xticks(range(0, len(per_min), step))
    ax.set_xticklabels(per_min.index[::step].strftime("%H:%M"), rotation=0, fontsize=8)
    ax.set_yticks([])
    ax.set_title("Detection Density Over Time (per minute)")
    ax.set_xlabel("Time")
    fig.colorbar(im, ax=ax, label="Detections", shrink=0.9)
    fig.tight_layout()
    _save(fig, "detection_heatmap.png")


def ocr_conf_by_state(df: pd.DataFrame) -> None:
    """Chart 3: OCR confidence box plot grouped by the plate's state code."""
    df = df[df["plate_text"].str.len() >= 2].copy()
    if df.empty:
        print("   ⏭️  skipped (no detection data)")
        return
    df["state"] = df["plate_text"].str[:2].str.upper()
    top_states = df["state"].value_counts().head(8).index
    df = df[df["state"].isin(top_states)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    data = [g["ocr_confidence"].values for _, g in df.groupby("state")]
    labels = sorted(df["state"].unique())
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor(C_PRIMARY)
        patch.set_alpha(0.75)
    ax.axhline(0.4, color=C_ALERT, linestyle="--", linewidth=1.2, label="OCR threshold (0.4)")
    ax.set_title("OCR Confidence by Plate State Code")
    ax.set_xlabel("State code")
    ax.set_ylabel("OCR confidence")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "ocr_conf_by_state.png")


def latency_breakdown(video: str, frames: int = 100) -> None:
    """Chart 4: stacked YOLO vs preprocess+OCR time per frame."""
    if not MODEL_PATH.exists():
        print("   ⏭️  skipped (models/best.pt missing — add weights first)")
        return
    import cv2
    from src.ocr import PlateOCR
    from src.preprocess import preprocess_plate
    from src.utils import load_config

    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"   ⏭️  skipped (video '{video}' could not be opened)")
        return
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    device = cfg["detection"].get("device", "cpu")
    ocr = PlateOCR(cfg)

    t_detect = t_ocr = 0.0
    used = 0
    while used < frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        dets = model.predict(frame, conf=cfg["detection"]["confidence_threshold"],
                             device=device, verbose=False)
        t_detect += time.perf_counter() - t0
        for det in dets:
            for box in det.boxes.xyxy.int().tolist():
                x1, y1, x2, y2 = box
                crop = det.orig_img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                t1 = time.perf_counter()
                ocr.recognize(preprocess_plate(crop, cfg))
                t_ocr += time.perf_counter() - t1
        used += 1
    cap.release()
    if used == 0:
        print("   ⏭️  skipped (no frames could be read)")
        return

    det_ms = t_detect / used * 1000
    # OCR time is per plate, not per frame — average plates per frame for stacking
    ocr_ms = t_ocr / used * 1000
    total = det_ms + ocr_ms

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(["per-frame latency"], [det_ms], color=C_PRIMARY, label="YOLO detection")
    ax.barh(["per-frame latency"], [ocr_ms], left=[det_ms], color=C_ACCENT, label="Preprocess + OCR")
    ax.text(det_ms / 2, 0, f"{det_ms:.1f} ms", va="center", ha="center",
            color="white", fontweight="bold")
    ax.text(det_ms + ocr_ms / 2, 0, f"{ocr_ms:.1f} ms", va="center", ha="center",
            color="white", fontweight="bold")
    ax.axvline(150, color=C_ALERT, linestyle="--", linewidth=1.2, label="Target (150 ms)")
    ax.set_title(f"Latency Breakdown — total {total:.1f} ms (~{1 / (total / 1000):.1f} FPS)")
    ax.set_xlabel("milliseconds")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, "latency_breakdown.png")


def main():
    parser = argparse.ArgumentParser(description="Generate ALPR performance benchmark charts")
    parser.add_argument("--video", default=None,
                        help="Video file for FPS/latency benchmarks (default: config video.source)")
    parser.add_argument("--frames", type=int, default=300,
                        help="Frames to benchmark (default: 300)")
    args = parser.parse_args()

    cfg = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text())
    video = args.video or cfg["video"]["source"]

    df = pd.DataFrame()
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

    print("📊 ALPR benchmark charts → outputs/visualizations/\n")
    print("1. FPS vs confidence threshold:")
    fps_vs_threshold(video, args.frames)
    print("2. Detection density heatmap:")
    detection_heatmap(df)
    print("3. OCR confidence by state code:")
    ocr_conf_by_state(df)
    print("4. Latency breakdown:")
    latency_breakdown(video)
    print("\nDone.")


if __name__ == "__main__":
    main()
