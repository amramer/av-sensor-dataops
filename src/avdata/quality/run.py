"""QUALITY stage: run schemas + checks, write reports, fail on ERRORs.

Outputs
  reports/quality/summary.json     DVC metric: counts per check and pass rates
  reports/quality/issues.parquet   every issue, one row each (for dashboards)
  data/silver/frames_qc.parquet    per-frame verdict used by curation
  data/silver/annotations_qc.parquet  per-box label quality
"""

from __future__ import annotations

import pandas as pd

from avdata import paths
from avdata.config import Params
from avdata.quality import checks, schemas
from avdata.utils import get_logger, read_parquet, write_json, write_parquet

log = get_logger(__name__)

ISSUE_COLUMNS = [
    "check",
    "severity",
    "entity_type",
    "entity_id",
    "scene_name",
    "channel",
    "value",
    "message",
]


class QualityGateError(RuntimeError):
    pass


def load_tables() -> checks.Tables:
    return checks.Tables(
        frames=read_parquet(paths.FRAMES),
        features=read_parquet(paths.FEATURES),
        annotations=read_parquet(paths.ANNOTATIONS),
        sync=read_parquet(paths.SYNC),
    )


def evaluate(t: checks.Tables, p: Params) -> pd.DataFrame:
    records: list[dict] = []
    records += schemas.validate(schemas.FRAMES, t.frames)
    records += schemas.validate(schemas.ANNOTATIONS, t.annotations)
    records += schemas.validate(schemas.FEATURES, t.features)
    for check in checks.CHECKS:
        found = check(t, p)
        log.info("  %-26s %5d issue(s)", check.__name__, len(found))
        records += found
    issues = pd.DataFrame(records, columns=ISSUE_COLUMNS)
    # explicit types, so an empty or all-null column still has a Parquet type Spark can read
    issues = issues.astype(
        {c: "string" for c in ISSUE_COLUMNS if c != "value"} | {"value": "float64"}
    )
    return issues.sort_values(["severity", "check", "entity_id"], na_position="last").reset_index(
        drop=True
    )


def summarize(
    t: checks.Tables, issues: pd.DataFrame, frames_qc: pd.DataFrame, labels_qc: pd.DataFrame
) -> dict:
    by_check = (
        issues.groupby(["check", "severity"]).size().rename("n").reset_index()
        if len(issues)
        else pd.DataFrame(columns=["check", "severity", "n"])
    )
    per_channel = frames_qc.groupby("channel")["qc_pass"].mean().round(4).to_dict()
    return {
        "errors": int((issues["severity"] == checks.ERROR).sum()),
        "warnings": int((issues["severity"] == checks.WARN).sum()),
        "issues_by_check": {f"{r.check}.{r.severity}": int(r.n) for r in by_check.itertuples()},
        "frames_total": len(frames_qc),
        "frame_pass_rate": round(float(frames_qc["qc_pass"].mean()), 4),
        "frame_pass_rate_by_channel": per_channel,
        "labels_total": len(labels_qc),
        "label_ok_rate": round(float((labels_qc["label_quality"] == "ok").mean()), 4)
        if len(labels_qc)
        else None,
    }


def run(p: Params) -> dict:
    t = load_tables()
    log.info("running quality checks")
    issues = evaluate(t, p)
    frames_qc = checks.frame_status(t, issues)
    labels_qc = checks.label_status(t, issues, p)
    summary = summarize(t, issues, frames_qc, labels_qc)

    write_parquet(issues, paths.QUALITY_REPORTS / "issues.parquet")
    write_parquet(frames_qc, paths.FRAMES_QC)
    write_parquet(labels_qc, paths.ANNOTATIONS_QC)
    write_json(summary, paths.QUALITY_REPORTS / "summary.json")
    log.info(
        "quality: %d errors, %d warnings, frame pass rate %.1f%%",
        summary["errors"],
        summary["warnings"],
        100 * summary["frame_pass_rate"],
    )

    if summary["errors"] > p.quality.max_error_issues:
        top = issues[issues["severity"] == checks.ERROR].head(10)["message"].tolist()
        raise QualityGateError(
            f"{summary['errors']} ERROR issue(s) > allowed {p.quality.max_error_issues}. "
            f"First ones: {top}. Full list: reports/quality/issues.parquet"
        )
    return summary
