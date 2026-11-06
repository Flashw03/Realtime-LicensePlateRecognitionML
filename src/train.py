"""
YOLO License Plate Detection Training Script.

Primary usage: Google Colab (T4 16GB)
Fallback: Local RTX 4050 6GB (reduced batch size)

Dataset: Kaggle "Indian Number Plate Annotation" by Sunraj Bishnolia
https://www.kaggle.com/datasets/sunrajbishnolia/license-plate-detection

Usage:
    # On Colab (recommended):
    python src/train.py --colab --data-path /content/data/data.yaml

    # Local (fallback, reduced performance):
    python src/train.py --data-path data/raw/data.yaml

    # Local + download the Kaggle dataset first (needs kaggle.json — see below):
    python src/train.py --download

Kaggle CLI auth: place your API token at ~/.kaggle/kaggle.json
(Linux/Colab) or C:\\Users\\<user>\\.kaggle\\kaggle.json (Windows).
Get it from Kaggle → Profile → Settings → API → Create New Token.
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
KAGGLE_DATASET = "sunrajbishnolia/license-plate-detection"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.yaml"


def detect_gpu():
    """Return (has_cuda, gpu_name, vram_gb)."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            # Round: 6140 MiB is 5.997 GiB — a "6GB" card must not fall
            # through the >= 6GB tier check on a float rounding artifact
            return True, props.name, round(props.total_memory / 1024 ** 3)
    except ImportError:
        pass
    return False, None, 0.0


def select_training_config(colab: bool, vram_gb: float, cfg: dict) -> dict:
    """Pick model size and batch size based on --colab flag / local VRAM."""
    training_cfg = cfg.get("training", {})
    if colab:
        model, batch = "yolov8s.pt", 16
        reason = "--colab mode (T4 16GB)"
    elif vram_gb >= 12:
        model, batch = "yolov8s.pt", 16
        reason = f"local VRAM {vram_gb:.1f}GB ≥ 12GB"
    elif vram_gb >= 6:
        model, batch = "yolov8n.pt", 8
        reason = f"local VRAM {vram_gb:.1f}GB ≥ 6GB"
    else:
        model, batch = "yolov8n.pt", 4
        reason = "CPU only / < 6GB VRAM"
    return {
        "model": model,
        "batch": batch,
        "epochs": int(training_cfg.get("epochs", 80)),
        "imgsz": int(training_cfg.get("img_size", 640)),
        "amp": bool(training_cfg.get("amp", True)),
        "patience": int(training_cfg.get("patience", 10)),
        "reason": reason,
    }


def dataset_root(data_path: Path) -> Path:
    """Directory that holds train/ valid/ test/ (parent of data.yaml)."""
    return data_path.parent


def dataset_already_downloaded(root: Path) -> bool:
    """True if the dataset root already has train/valid/test images on disk."""
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        if img_dir.is_dir() and any(p.is_file() for p in img_dir.iterdir()):
            return True
    return False


def kaggle_token_path() -> Path:
    """Location of the Kaggle API token for this OS."""
    return Path.home() / ".kaggle" / "kaggle.json"


def download_dataset(root: Path) -> None:
    """Download + extract the Kaggle dataset into `root` via the Kaggle CLI."""
    if not kaggle_token_path().exists():
        sys.exit(
            f"❌ Kaggle API token not found at '{kaggle_token_path()}'.\n"
            "   Kaggle → Profile → Settings → API → Create New Token, then save\n"
            "   the downloaded kaggle.json to that path."
        )

    root.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Downloading Kaggle dataset '{KAGGLE_DATASET}' into {root} ...")
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
           "-d", KAGGLE_DATASET, "-p", str(root), "--unzip"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit("❌ Kaggle download failed — check the messages above.")


def verify_dataset_structure(root: Path) -> None:
    """Check the expected train/valid/test layout; warn if it differs."""
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            print(f"⚠️  Expected '{img_dir}' and '{lbl_dir}' — not found. "
                  "Check the extracted Kaggle folder structure.")


def create_data_yaml_if_missing(data_path: Path) -> None:
    """Write the dataset data.yaml if it doesn't exist yet."""
    if data_path.exists():
        return
    root = dataset_root(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w") as f:
        f.write(
            f"path: {root.as_posix()}\n"
            "train: train/images\n"
            "val: valid/images\n"
            "test: test/images\n"
            "\n"
            "nc: 1\n"
            "names: ['license_plate']\n"
        )
    print(f"📝 Created {data_path}")


def print_class_names(data_yaml: Path) -> None:
    """Log dataset class names — catches train/dataset class mismatches early."""
    with open(data_yaml, "r") as f:
        meta = yaml.safe_load(f)
    names = meta.get("names", {})
    print(f"🏷️  Dataset classes: {names}")


def copy_best_weights() -> Path:
    """Copy the newest runs/**/best.pt to models/best.pt and return its path."""
    runs = sorted(glob.glob(str(PROJECT_ROOT / "runs" / "**" / "best.pt"), recursive=True),
                  key=os.path.getmtime)
    if not runs:
        sys.exit("❌ No best.pt found under runs/ — training may have failed.")
    best = Path(runs[-1])
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / "best.pt"
    shutil.copy2(best, dest)
    return best


def print_final_metrics(run_dir: Path) -> None:
    """Print the last-epoch metrics from the run's results.csv."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        print("ℹ️  No results.csv found — skipping metrics summary.")
        return
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    last = rows[-1]

    def pick(*keys):
        for k in keys:
            for name, val in last.items():
                if name and name.strip() == k:
                    return val
        return "n/a"

    print("\n📊 Final training metrics (last epoch):")
    print(f"   precision : {pick('metrics/precision(B)')}")
    print(f"   recall    : {pick('metrics/recall(B)')}")
    print(f"   mAP@50    : {pick('metrics/mAP50(B)')}")
    print(f"   mAP@50-95 : {pick('metrics/mAP50-95(B)')}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 license plate training (local/Colab)")
    parser.add_argument("--colab", action="store_true",
                        help="Colab mode: yolov8s, batch=16")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH),
                        help="Path to the dataset data.yaml "
                             "(default: data/raw/data.yaml; on Colab: /content/data/data.yaml)")
    parser.add_argument("--download", action="store_true",
                        help="Download the Kaggle dataset first (needs kaggle.json)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    args = parser.parse_args()

    # GPU report
    has_cuda, gpu_name, vram_gb = detect_gpu()
    if has_cuda:
        print(f"🎮 GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("🎮 GPU: none detected — training on CPU (very slow; prefer Colab).")

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    train_cfg = select_training_config(args.colab, vram_gb, cfg)
    if args.epochs:
        train_cfg["epochs"] = args.epochs

    print(f"🧠 Selected config: model={train_cfg['model']} batch={train_cfg['batch']} "
          f"epochs={train_cfg['epochs']} imgsz={train_cfg['imgsz']} "
          f"amp={train_cfg['amp']} ({train_cfg['reason']})")

    # Dataset: optional Kaggle download, then data.yaml resolution
    data_path = Path(args.data_path)
    root = dataset_root(data_path)
    if args.download:
        if dataset_already_downloaded(root):
            print("📦 Existing dataset found — skipping download.")
        else:
            download_dataset(root)
    elif not data_path.exists() and not dataset_already_downloaded(root):
        print(
            f"💡 No dataset at '{root}'. Run with --download to fetch the Kaggle\n"
            "   dataset, or pass --data-path pointing at an existing data.yaml."
        )

    if not data_path.exists():
        create_data_yaml_if_missing(data_path)
    if not data_path.exists():
        sys.exit(f"❌ data.yaml not found at '{data_path}' — cannot train.")
    if not dataset_already_downloaded(root):
        sys.exit(f"❌ No images found under '{root}' — cannot train. Use --download.")

    verify_dataset_structure(root)
    print_class_names(data_path)
    print(f"📁 Using dataset config: {data_path}")

    # Train
    from ultralytics import YOLO
    model = YOLO(train_cfg["model"])
    model.train(
        data=str(data_path),
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch"],
        imgsz=train_cfg["imgsz"],
        amp=train_cfg["amp"],
        patience=train_cfg["patience"],
        project=str(PROJECT_ROOT / "runs"),
        name="lp_train",
        exist_ok=True,
    )

    best = copy_best_weights()
    print(f"✅ Best weights copied: {best} → {MODELS_DIR / 'best.pt'}")
    print_final_metrics(best.parent)


if __name__ == "__main__":
    main()
