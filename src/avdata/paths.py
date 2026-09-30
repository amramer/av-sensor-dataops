"""Standard locations of every pipeline layer (medallion layout).

raw     read-only sensor logs, exactly as delivered (DVC-tracked, never modified)
bronze  raw metadata tables converted 1:1 to typed Parquet
silver  cleaned, joined, feature-enriched tables (one row per sensor frame)
gold    curated datasets: scenario tags, splits, 2D labels
ml_ready  training-framework format (YOLO) built from gold
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(".")  # relative: every stage runs from the repo root
DATA = ROOT / "data"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
ML_READY = DATA / "ml_ready"
REPORTS = ROOT / "reports"
QUALITY_REPORTS = REPORTS / "quality"
MODELS = ROOT / "models"
METRICS = ROOT / "metrics"

# silver
FRAMES = SILVER / "frames.parquet"
ANNOTATIONS = SILVER / "annotations.parquet"
SYNC = SILVER / "sync.parquet"
FEATURES = SILVER / "frame_features.parquet"
FRAMES_QC = SILVER / "frames_qc.parquet"
ANNOTATIONS_QC = SILVER / "annotations_qc.parquet"

# gold
SAMPLES = GOLD / "samples.parquet"
LABELS_2D = GOLD / "labels_2d.parquet"

# ml_ready
YOLO_DIR = ML_READY / "yolo"


def ensure(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
