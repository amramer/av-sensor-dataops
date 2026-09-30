"""EXTRACT: raw nuScenes metadata (JSON) -> bronze Parquet tables.

Bronze is a 1:1 typed copy of the source tables. Nothing is filtered or
joined here, so any later stage can be rebuilt from bronze without touching
the raw release again. A manifest records row counts and the SHA-256 of each
source file, which is the start of the lineage chain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from avdata import paths
from avdata.config import Params
from avdata.utils import get_logger, sha256_file, write_json, write_parquet

log = get_logger(__name__)

# Tables the pipeline needs, with the columns each one must provide.
REQUIRED_TABLES: dict[str, list[str]] = {
    "scene": ["token", "log_token", "name", "description", "first_sample_token"],
    "sample": ["token", "timestamp", "scene_token"],
    "sample_data": [
        "token",
        "sample_token",
        "ego_pose_token",
        "calibrated_sensor_token",
        "timestamp",
        "is_key_frame",
        "filename",
        "width",
        "height",
    ],
    "sensor": ["token", "channel", "modality"],
    "calibrated_sensor": ["token", "sensor_token", "translation", "rotation", "camera_intrinsic"],
    "ego_pose": ["token", "timestamp", "translation", "rotation"],
    "log": ["token", "location", "vehicle", "date_captured"],
    "sample_annotation": [
        "token",
        "sample_token",
        "instance_token",
        "visibility_token",
        "attribute_tokens",
        "translation",
        "size",
        "rotation",
        "num_lidar_pts",
        "num_radar_pts",
    ],
    "instance": ["token", "category_token"],
    "category": ["token", "name"],
    "visibility": ["token", "level"],
    "attribute": ["token", "name"],
}


class SourceSchemaError(ValueError):
    pass


def read_table(tables_dir: Path, name: str) -> pd.DataFrame:
    path = tables_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing nuScenes table {path}")
    df = pd.DataFrame(json.loads(path.read_text()))
    missing = set(REQUIRED_TABLES[name]) - set(df.columns)
    if missing:
        raise SourceSchemaError(f"{name}.json is missing columns {sorted(missing)}")
    return df


def run(p: Params) -> dict:
    tables_dir = p.data.tables_dir
    log.info("extracting nuScenes %s tables from %s", p.data.version, tables_dir)
    # no timestamps in outputs: identical inputs must give byte-identical outputs
    manifest: dict = {"source_version": p.data.version, "tables": {}}
    for name in REQUIRED_TABLES:
        df = read_table(tables_dir, name)
        if "token" in df.columns and df["token"].duplicated().any():
            raise SourceSchemaError(f"{name}: duplicate primary keys (token)")
        write_parquet(df, paths.BRONZE / f"{name}.parquet")
        manifest["tables"][name] = {
            "rows": len(df),
            "sha256": sha256_file(tables_dir / f"{name}.json"),
        }
        log.info("  %-18s %8d rows", name, len(df))
    write_json(manifest, paths.BRONZE / "_manifest.json")
    return manifest
