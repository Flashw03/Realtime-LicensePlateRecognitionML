"""Streamlit dashboard for the ALPR pipeline.

Reads detections from outputs/logs/detections.csv (or an uploaded CSV)
and presents: headline metrics, detections-over-time, the detection log,
confidence distribution, and top plates.

Run with:
    streamlit run dashboard/app.py
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml

# ── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ALPR Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_CSV = Path("outputs/logs/detections.csv")
CONFIG_PATH = Path("config/config.yaml")

# Muted palette (matches the project design spec)
C_PRIMARY = "#4a90d9"
C_ACCENT = "#2ecc71"
C_ALERT = "#e74c3c"
C_TEXT = "#333333"


def load_refresh_interval() -> int:
    """Refresh interval (seconds) from config, with a safe fallback."""
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return int(cfg.get("dashboard", {}).get("refresh_interval_seconds", 2))
    except (OSError, ValueError):
        return 2


def load_ocr_threshold() -> float:
    """OCR confidence threshold from config, for the distribution marker."""
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return float(cfg.get("ocr", {}).get("confidence_threshold", 0.4))
    except (OSError, ValueError):
        return 0.4


def _source_mtime(source) -> float | None:
    """Mtime of a file-path source (None for uploads) — busts the cache
    when the pipeline appends new rows."""
    try:
        return Path(source).stat().st_mtime
    except OSError:
        return None


@st.cache_data
def load_data(source, mtime: float | None = None) -> pd.DataFrame:
    """Load and normalize a detections CSV (path or raw bytes).

    `mtime` participates in the cache key so new rows appended by the
    pipeline are picked up on the next rerun.
    """
    if isinstance(source, (bytes, bytearray)):
        df = pd.read_csv(pd.io.common.BytesIO(source))
    else:
        df = pd.read_csv(source)
    expected = ["timestamp", "frame_no", "plate_text", "ocr_confidence",
                "det_confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    df = df[[c for c in expected if c in df.columns]].copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
    for col in ("ocr_confidence", "det_confidence"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ocr_confidence", "det_confidence"])
    return df


# ── Sidebar controls ─────────────────────────────────────────────────────
st.sidebar.header("Controls")

uploaded = st.sidebar.file_uploader(
    "📁 Upload a detections CSV", type=["csv"],
    help="Leave empty to auto-read outputs/logs/detections.csv",
)
if uploaded is not None:
    source = uploaded.getvalue()
    source_label = uploaded.name
    df_all = load_data(source)
else:
    source = DEFAULT_CSV
    source_label = str(DEFAULT_CSV)
    df_all = load_data(str(DEFAULT_CSV), mtime=_source_mtime(DEFAULT_CSV))

auto_refresh = st.sidebar.toggle("🔄 Auto-refresh", value=False)
default_interval = load_refresh_interval()
refresh_interval = st.sidebar.slider(
    "Refresh interval (seconds)", 1, 30, default_interval,
    disabled=not auto_refresh,
)

min_conf = st.sidebar.slider(
    "🎚️ Minimum OCR confidence", 0.0, 1.0, 0.0, 0.05,
    help="Filter the log to detections at or above this confidence",
)

date_from, date_to = None, None
if not df_all.empty:
    dates = df_all["timestamp"].dt.date
    date_range = st.sidebar.date_input(
        "📅 Date filter",
        value=(dates.min(), dates.max()),
        min_value=dates.min(),
        max_value=dates.max(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        date_from, date_to = date_range
    elif isinstance(date_range, tuple) and len(date_range) == 1:
        date_from = date_to = date_range[0]

st.sidebar.caption(f"Data source: `{source_label}`")

# ── Empty state ──────────────────────────────────────────────────────────
if df_all.empty:
    st.info(
        "**No detections yet.** Run the pipeline first:\n\n"
        "```bash\npython scripts/run.py\n```"
    )
    st.stop()

# ── Filtering ────────────────────────────────────────────────────────────
df = df_all[df_all["ocr_confidence"] >= min_conf].copy()
if date_from is not None:
    df = df[df["timestamp"].dt.date >= date_from]
if date_to is not None:
    df = df[df["timestamp"].dt.date <= date_to]

if df.empty:
    st.warning("No detections match the current filters — relax the confidence "
               "slider or date range in the sidebar.")
    st.stop()

# ── Header ───────────────────────────────────────────────────────────────
st.title("🚗 License Plate Recognition Dashboard")
st.caption("Real-time detection & OCR results · Indian plate format")

# ── Metric cards ─────────────────────────────────────────────────────────
total_detections = len(df)
unique_plates = df["plate_text"].nunique()
avg_ocr_conf = df["ocr_confidence"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total Detections", f"{total_detections:,}")
col2.metric("Unique Plates", f"{unique_plates:,}")
col3.metric("Avg OCR Confidence", f"{avg_ocr_conf:.3f}")

# ── Detections over time ─────────────────────────────────────────────────
st.subheader("📊 Detections Over Time")
per_minute = (
    df.set_index("timestamp")
    .resample("1min")
    .size()
    .rename("detections")
)
st.line_chart(per_minute, color=C_PRIMARY, height=240)

# ── Detection log table ──────────────────────────────────────────────────
st.subheader("📋 Detection Log")

table = df.sort_values("timestamp", ascending=False).head(500).copy()
table["Time"] = table["timestamp"].dt.strftime("%H:%M:%S")
styled = (
    table[["Time", "frame_no", "plate_text", "ocr_confidence", "det_confidence"]]
    .rename(columns={
        "frame_no": "Frame",
        "plate_text": "Plate",
        "ocr_confidence": "OCR Conf",
        "det_confidence": "Det Conf",
    })
    .style.map(
        lambda v: "background-color: #eafaf1; color: #1e7e46;"
        if v > 0.9 else "",
        subset=["OCR Conf", "Det Conf"],
    )
    .format({"OCR Conf": "{:.1%}", "Det Conf": "{:.1%}"})
)
st.dataframe(styled, use_container_width=True, height=320)

# ── Confidence distribution + Top plates ────────────────────────────────
col_charts, col_top = st.columns([1, 1], gap="large")

with col_charts:
    st.subheader("📈 Confidence Distribution")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.hist(df["ocr_confidence"], bins=20, range=(0.0, 1.0),
            color=C_PRIMARY, edgecolor="white", linewidth=0.6)
    threshold = load_ocr_threshold()
    ax.axvline(threshold, color=C_ALERT, linestyle="--", linewidth=1.4,
               label=f"Threshold ({threshold:.2f})")
    ax.set_xlabel("OCR confidence")
    ax.set_ylabel("Detections")
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

with col_top:
    st.subheader("🥧 Top 10 Plates")
    top = df["plate_text"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.barh(top.index[::-1], top.values[::-1], color=C_PRIMARY, height=0.62)
    ax.set_xlabel("Detections")
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

# ── Auto-refresh ─────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
