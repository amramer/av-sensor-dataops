"""Typed access to params.yaml.

Every stage reads its settings through `load_params()`, so a typo in
params.yaml fails fast with a clear validation error instead of deep
inside a stage.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

PARAMS_FILE = Path(os.environ.get("AVDATA_PARAMS", "params.yaml"))


class DataParams(BaseModel):
    raw_dir: Path
    version: str = "v1.0-mini"
    scenes: Literal["all"] | list[str] = "all"

    @property
    def tables_dir(self) -> Path:
        return self.raw_dir / self.version


class SensorParams(BaseModel):
    enabled: list[str]
    reference_channel: str = "LIDAR_TOP"
    nominal_hz: dict[str, float]

    @field_validator("enabled")
    @classmethod
    def _not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one sensor channel must be enabled")
        return v


class EtlParams(BaseModel):
    include_sweeps: bool = True
    workers: int = Field(4, ge=1)
    feature_engine: Literal["pandas", "spark"] = "pandas"
    image_feature_width: int = Field(320, ge=32)


class QualityParams(BaseModel):
    sync_tolerance_ms: float = 50
    gap_factor: float = 2.5
    min_lidar_points: int = 5000
    min_box_lidar_points: int = 1
    night_brightness_max: float = 90
    projection_ratio_tolerance: float = 1.3
    max_error_issues: int = 0


class CurationParams(BaseModel):
    seed: int = 42
    camera: str = "CAM_FRONT"
    val_fraction: float = Field(0.2, ge=0, lt=1)
    test_fraction: float = Field(0.2, ge=0, lt=1)
    stratify_by: str = "time_of_day"
    min_visibility: int = Field(2, ge=1, le=4)
    min_box_px: int = 6
    speed_bins_mps: list[float] = [0.5, 5.0]
    crowded_min_pedestrians: int = 5
    regression_tags: list[str] = []
    classes: dict[str, str]

    @property
    def class_names(self) -> list[str]:
        """Detection classes in a stable order (first appearance in the mapping)."""
        return list(dict.fromkeys(self.classes.values()))


class TrainParams(BaseModel):
    model: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    seed: int = 42
    device: str = "auto"
    workers: int = 4
    fraction: float = Field(1.0, gt=0, le=1)


class GateParams(BaseModel):
    min_map50: float = 0.25
    max_slice_drop: float = 0.10


class FusionParams(BaseModel):
    lidar_channel: str = "LIDAR_TOP"
    score_threshold: float = 0.25
    box_shrink: float = Field(0.1, ge=0, lt=0.9)
    ground_offset_m: float = 0.3
    min_points: int = Field(3, ge=1)
    cluster_gap_m: float = Field(0.8, gt=0)
    near_percentile: float = Field(10, ge=0, le=100)
    eval_thresholds_m: list[float] = [0.5, 1.0, 2.0, 4.0]
    tp_threshold_m: float = 2.0
    class_range_m: dict[str, float] = {}


class VizParams(BaseModel):
    samples: int = 12
    failures: int = 6
    bev_range_m: float = 60
    demo_samples: int = 16


class KpiParams(BaseModel):
    db_url: str = "sqlite:///reports/kpi.db"

    @property
    def resolved_url(self) -> str:
        return os.environ.get("AVDATA_KPI_DB_URL", self.db_url)


class Params(BaseModel):
    data: DataParams
    sensors: SensorParams
    etl: EtlParams
    quality: QualityParams
    curation: CurationParams
    train: TrainParams
    gate: GateParams
    fusion: FusionParams = FusionParams()
    viz: VizParams = VizParams()
    kpi: KpiParams


def load_params(path: Path | str | None = None) -> Params:
    path = Path(path) if path else PARAMS_FILE
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Params.model_validate(raw)


@lru_cache(maxsize=1)
def params() -> Params:
    """Cached params for the current process."""
    return load_params()
