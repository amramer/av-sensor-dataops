"""Publish KPIs to a SQL store (SQLite locally, Postgres in docker compose).

This is the history that Grafana charts: every pipeline run appends one
snapshot, so data growth, quality trends and model performance over dataset
versions become time series. Tables are long/narrow on purpose, so adding a
KPI never needs a schema migration.

  kpi_history     run_id, published_at, git_commit, dataset_version, metric, value
  quality_issues  run_id, published_at, check, severity, scene_name, channel, message
  coverage        run_id, published_at, location, time_of_day, weather, speed_bin, split,
                  samples, ml_ready
  model_metrics   run_id, published_at, model_version, slice, metric, value
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from avdata import paths
from avdata.config import Params
from avdata.utils import get_logger, git_commit, utc_now

log = get_logger(__name__)


def flatten(d: dict, prefix: str = "") -> dict[str, float]:
    """{'a': {'b': 1}} -> {'a.b': 1.0}; keeps only numeric leaves."""
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out |= flatten(v, key)
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
            out[key] = float(v)
    return out


def dataset_version(kpis_file: Path) -> str:
    """Content fingerprint of the dataset snapshot the KPIs describe."""
    return hashlib.sha256(kpis_file.read_bytes()).hexdigest()[:12]


def run(p: Params) -> dict:
    url = p.kpi.resolved_url
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url)
    kpis_file = paths.REPORTS / "kpis.json"
    kpis = json.loads(kpis_file.read_text())
    meta = {"run_id": uuid.uuid4().hex[:12], "published_at": pd.Timestamp(utc_now())}

    hist = pd.DataFrame([{"metric": k, "value": v} for k, v in flatten(kpis).items()]).assign(
        **meta, git_commit=git_commit(), dataset_version=dataset_version(kpis_file)
    )
    issues = pd.read_parquet(paths.QUALITY_REPORTS / "issues.parquet")[
        ["check", "severity", "scene_name", "channel", "message"]
    ].assign(**meta)
    coverage = pd.read_parquet(paths.REPORTS / "coverage.parquet").assign(**meta)

    written = {}
    with engine.begin() as conn:
        for name, df in {
            "kpi_history": hist,
            "quality_issues": issues,
            "coverage": coverage,
        }.items():
            df.to_sql(name, conn, if_exists="append", index=False)
            written[name] = len(df)
        eval_file = paths.METRICS / "eval.json"
        if eval_file.exists():
            ev = json.loads(eval_file.read_text())
            rows = [
                {"slice": s, "metric": m, "value": v}
                for s, ms in ev.get("slices", {}).items()
                for m, v in ms.items()
                if isinstance(v, (int, float))
            ]
            mm = pd.DataFrame(rows).assign(**meta, model_version=ev.get("model_version", "unknown"))
            mm.to_sql("model_metrics", conn, if_exists="append", index=False)
            written["model_metrics"] = len(mm)
    safe_url = url.split("@")[-1]
    log.info("published KPIs to %s: %s", safe_url, written)
    return {"run_id": meta["run_id"], "store": safe_url, "rows": written}
