"""KPI stage (PySpark): data volume, quality, label, coverage and ML-readiness KPIs.

Everything here is a Spark SQL aggregation over the silver/gold Parquet
tables, so it runs unchanged on a laptop (local[*]) or a cluster when the
dataset grows to the full nuScenes release or to a real fleet.

Outputs
  reports/kpis.json          DVC metrics (flat, diff-able with `dvc metrics diff`)
  reports/coverage.parquet   scenario coverage matrix
  reports/coverage.html      interactive Plotly report (coverage + quality)
"""

from __future__ import annotations

import pandas as pd

from avdata import paths
from avdata.config import Params
from avdata.utils import get_logger, write_json, write_parquet

log = get_logger(__name__)


def _load(spark, name: str, path) -> None:
    spark.read.parquet(str(path)).createOrReplaceTempView(name)


def compute(spark) -> tuple[dict, pd.DataFrame]:
    """Run all KPI queries. Tables must be registered as temp views."""
    q = lambda sql: spark.sql(sql).toPandas()  # noqa: E731

    volume = q("""
        SELECT channel,
               COUNT(*)                                   AS frames,
               SUM(CAST(is_key_frame AS INT))             AS keyframes,
               SUM(GREATEST(file_size_bytes, 0))          AS bytes
        FROM frames GROUP BY channel ORDER BY channel""")
    duration = q("""
        SELECT SUM(dur) / 1e6 AS seconds FROM (
            SELECT scene_name, MAX(timestamp) - MIN(timestamp) AS dur
            FROM frames GROUP BY scene_name)""")
    rate = (
        q("""
        SELECT channel, COUNT(*) / ((MAX(timestamp) - MIN(timestamp)) / 1e6) AS hz
        FROM frames GROUP BY scene_name, channel""")
        .groupby("channel")["hz"]
        .median()
    )
    qc = q("""
        SELECT channel, AVG(CAST(qc_pass AS DOUBLE)) AS pass_rate
        FROM frames_qc GROUP BY channel""")
    sync = q("""
        SELECT channel,
               PERCENTILE_APPROX(ABS(offset_ms), 0.5)  AS p50,
               PERCENTILE_APPROX(ABS(offset_ms), 0.95) AS p95,
               MAX(ABS(offset_ms))                     AS max
        FROM sync WHERE has_keyframe GROUP BY channel""")
    issues = q("SELECT `check`, severity, COUNT(*) AS n FROM issues GROUP BY `check`, severity")
    labels = q("""
        SELECT label_quality, COUNT(*) AS n FROM annotations_qc
        WHERE det_class IS NOT NULL GROUP BY label_quality""")
    boxes = q(
        "SELECT det_class, distance_bin, COUNT(*) AS n FROM labels_2d "
        "GROUP BY det_class, distance_bin"
    )
    readiness = q("""
        SELECT split, COUNT(*) AS samples, SUM(CAST(ml_ready AS INT)) AS ready,
               SUM(CAST(in_regression_set AND ml_ready AS INT)) AS regression
        FROM samples GROUP BY split""")
    train_boxes = q("""
        SELECT l.det_class, COUNT(*) AS n FROM labels_2d l
        JOIN samples s ON l.sample_token = s.sample_token
        WHERE s.split = 'train' GROUP BY l.det_class""")
    coverage = q("""
        SELECT location, time_of_day, weather, speed_bin, split,
               COUNT(*) AS samples, SUM(CAST(ml_ready AS INT)) AS ml_ready
        FROM samples GROUP BY location, time_of_day, weather, speed_bin, split""")

    total_labels = int(labels["n"].sum())
    lab = labels.set_index("label_quality")["n"].to_dict()
    kpis: dict = {
        "volume": {
            "bytes_total": int(volume["bytes"].sum()),
            "gb_total": round(float(volume["bytes"].sum()) / 1e9, 4),
            "frames_total": int(volume["frames"].sum()),
            "recording_seconds": round(float(duration["seconds"].iloc[0] or 0), 2),
            "by_channel": {
                r.channel: {
                    "frames": int(r.frames),
                    "keyframes": int(r.keyframes),
                    "mb": round(r.bytes / 1e6, 2),
                    "measured_hz": round(float(rate[r.channel]), 2),
                }
                for r in volume.itertuples()
            },
        },
        "quality": {
            "frame_pass_rate_by_channel": {
                r.channel: round(r.pass_rate, 4) for r in qc.itertuples()
            },
            "sync_abs_offset_ms": {
                r.channel: {"p50": round(r.p50, 2), "p95": round(r.p95, 2), "max": round(r.max, 2)}
                for r in sync.itertuples()
            },
            "issues": {f"{r.check}.{r.severity}": int(r.n) for r in issues.itertuples()},
        },
        "labels": {
            "boxes_3d": total_labels,
            "ok_rate": round(lab.get("ok", 0) / total_labels, 4) if total_labels else None,
            "weak_rate": round(lab.get("weak", 0) / total_labels, 4) if total_labels else None,
            "low_visibility_rate": round(lab.get("low_visibility", 0) / total_labels, 4)
            if total_labels
            else None,
            "boxes_2d_by_class": boxes.groupby("det_class")["n"].sum().astype(int).to_dict(),
            "boxes_2d_by_distance": boxes.groupby("distance_bin")["n"].sum().astype(int).to_dict(),
        },
        "readiness": {
            "samples_by_split": {r.split: int(r.samples) for r in readiness.itertuples()},
            "ml_ready_by_split": {r.split: int(r.ready) for r in readiness.itertuples()},
            "ml_ready_rate": round(float(readiness["ready"].sum() / readiness["samples"].sum()), 4),
            "regression_set_samples": int(readiness["regression"].sum()),
            "train_boxes_by_class": train_boxes.set_index("det_class")["n"].astype(int).to_dict(),
        },
    }
    cells = coverage.groupby(["location", "time_of_day", "weather"])["samples"].sum()
    full = pd.MultiIndex.from_product(cells.index.levels)
    kpis["coverage"] = {
        "cells_observed": int((cells > 0).sum()),
        "cells_possible": len(full),
        "empty_cells": ["/".join(c) for c in full if c not in cells.index],
    }
    return kpis, coverage


def missing_train_classes(kpis: dict, class_names: list[str]) -> list[str]:
    present = kpis["readiness"]["train_boxes_by_class"]
    return [c for c in class_names if present.get(c, 0) == 0]


def run(p: Params) -> dict:
    from avdata.kpi import report
    from avdata.spark_session import get_spark

    spark = get_spark("avdata-kpis")
    for name, path in {
        "frames": paths.FRAMES,
        "frames_qc": paths.FRAMES_QC,
        "sync": paths.SYNC,
        "annotations_qc": paths.ANNOTATIONS_QC,
        "samples": paths.SAMPLES,
        "labels_2d": paths.LABELS_2D,
        "issues": paths.QUALITY_REPORTS / "issues.parquet",
    }.items():
        _load(spark, name, path)
    kpis, coverage = compute(spark)
    kpis["readiness"]["train_classes_missing"] = missing_train_classes(kpis, p.curation.class_names)

    write_json(kpis, paths.REPORTS / "kpis.json")
    write_parquet(coverage, paths.REPORTS / "coverage.parquet")
    report.write_html(kpis, coverage, paths.REPORTS / "coverage.html")
    log.info(
        "KPIs: %.3f GB, %d frames, ml-ready %.1f%%, coverage %d/%d cells",
        kpis["volume"]["gb_total"],
        kpis["volume"]["frames_total"],
        100 * kpis["readiness"]["ml_ready_rate"],
        kpis["coverage"]["cells_observed"],
        kpis["coverage"]["cells_possible"],
    )
    return kpis
