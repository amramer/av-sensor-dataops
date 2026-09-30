"""FEATURES: decode every sensor file once and compute cheap per-frame stats.

This is the most expensive step (it touches every raw file), so it runs in
parallel. Two interchangeable engines produce the identical output table:

  pandas  process pool on one machine (default, fine for nuScenes mini)
  spark   PySpark job; set `spark.master` to a cluster to scale out to the
          full dataset with the same code

Decoding also doubles as the integrity check: a file that fails to decode
is recorded with its error and later becomes a `corrupt_file` issue.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from avdata import paths, sensors
from avdata.config import Params
from avdata.utils import get_logger, read_parquet, write_parquet

log = get_logger(__name__)

OUTPUT_COLUMNS = ["frame_token", "decode_ok", "decode_error", *sensors.FEATURE_COLUMNS]


def extract_one(task: tuple[str, str, str, int, int, int]) -> dict:
    """Features for a single file. Pure function, safe to run on any worker."""
    frame_token, path, modality, width, height, feat_w = task
    row: dict = {"frame_token": frame_token, "decode_ok": False, "decode_error": None}
    row.update({c: float("nan") for c in sensors.FEATURE_COLUMNS})
    p = Path(path)
    if not p.exists():
        row["decode_error"] = "missing file"
        return row
    ctx = {"width": width, "height": height, "image_feature_width": feat_w}
    try:
        row.update(sensors.get_modality(modality).features(p, ctx))
        row["decode_ok"] = True
    except sensors.CorruptFileError as e:
        row["decode_error"] = str(e)[:200]
    return row


def _tasks(frames: pd.DataFrame, p: Params) -> list[tuple]:
    raw = p.data.raw_dir
    return [
        (
            r.frame_token,
            str(raw / r.filename),
            r.modality,
            int(r.width or 0),
            int(r.height or 0),
            p.etl.image_feature_width,
        )
        for r in frames.itertuples()
    ]


def run_pandas(tasks: list[tuple], workers: int) -> pd.DataFrame:
    if workers <= 1:
        rows = [extract_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(extract_one, tasks, chunksize=32))
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def run_spark(tasks: list[tuple], workers: int) -> pd.DataFrame:
    from avdata.spark_session import get_spark

    spark = get_spark("avdata-features")
    rdd = spark.sparkContext.parallelize(tasks, numSlices=max(workers * 4, 1))
    rows = rdd.map(extract_one).collect()
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def run(p: Params) -> dict:
    frames = read_parquet(paths.FRAMES)
    tasks = _tasks(frames, p)
    workers = min(p.etl.workers, os.cpu_count() or 1)
    log.info(
        "extracting features for %d files with %s engine (%d workers)",
        len(tasks),
        p.etl.feature_engine,
        workers,
    )
    engine = run_spark if p.etl.feature_engine == "spark" else run_pandas
    feats = engine(tasks, workers).sort_values("frame_token").reset_index(drop=True)
    write_parquet(feats, paths.FEATURES)
    summary = {"files": len(feats), "decode_failures": int((~feats["decode_ok"]).sum())}
    log.info("features: %s", summary)
    return summary
