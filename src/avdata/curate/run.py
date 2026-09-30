"""CURATE stage: gold datasets from QC'd silver data.

Outputs
  data/gold/samples.parquet    one row per keyframe sample: scenario tags, split,
                               regression-set flag, ML-readiness and the reason if not
  data/gold/labels_2d.parquet  2D boxes on the curated camera
  reports/curation.json        DVC metric: dataset sizes, drop reasons
"""

from __future__ import annotations

import pandas as pd

from avdata import paths
from avdata.config import Params
from avdata.curate import labels2d, scenarios
from avdata.utils import get_logger, read_parquet, write_json, write_parquet

log = get_logger(__name__)


class SplitLeakageError(RuntimeError):
    pass


def check_no_leakage(samples: pd.DataFrame) -> None:
    per_scene = samples.groupby("scene_name")["split"].nunique()
    leaking = per_scene[per_scene > 1]
    if len(leaking):
        raise SplitLeakageError(f"scenes in more than one split: {list(leaking.index)}")


def run(p: Params) -> dict:
    frames = read_parquet(paths.FRAMES)
    annotations = read_parquet(paths.ANNOTATIONS)
    frames_qc = read_parquet(paths.FRAMES_QC)
    labels_qc = read_parquet(paths.ANNOTATIONS_QC)
    cam = p.curation.camera
    if cam not in p.sensors.enabled:
        raise ValueError(f"curation.camera={cam} must be in sensors.enabled")

    samples = scenarios.tag_samples(frames, annotations, p)
    samples["split"] = scenarios.assign_splits(samples, p)
    check_no_leakage(samples)

    # camera keyframe for each sample, with its QC verdict
    cam_frames = frames[frames["is_key_frame"] & (frames["channel"] == cam)].merge(
        frames_qc[["frame_token", "qc_pass", "qc_error", "sync_ok"]], on="frame_token"
    )
    samples = samples.merge(
        cam_frames[["sample_token", "frame_token", "filename", "qc_pass", "qc_error", "sync_ok"]],
        on="sample_token",
        how="left",
    )
    samples["not_ready_reason"] = None
    samples.loc[samples["frame_token"].isna(), "not_ready_reason"] = "no_camera_keyframe"
    samples.loc[samples["qc_error"].fillna(False).astype(bool), "not_ready_reason"] = (
        "camera_qc_error"
    )
    samples.loc[~samples["sync_ok"].fillna(True).astype(bool), "not_ready_reason"] = "out_of_sync"
    samples["ml_ready"] = samples["not_ready_reason"].isna()

    regression_tags = set(p.curation.regression_tags)
    samples["in_regression_set"] = (samples["split"] == "test") & samples["tags"].apply(
        lambda t: bool(regression_tags & set(t))
    )

    ready = cam_frames[
        cam_frames["sample_token"].isin(samples.loc[samples["ml_ready"], "sample_token"])
    ]
    labels, dropped = labels2d.project_labels(ready, annotations, labels_qc, p)
    samples = (
        samples.drop(columns=["qc_pass", "qc_error", "sync_ok"])
        .sort_values(["split", "scene_name", "sample_timestamp"])
        .reset_index(drop=True)
    )

    write_parquet(samples, paths.SAMPLES)
    write_parquet(labels, paths.LABELS_2D)
    summary = {
        "samples": len(samples),
        "ml_ready_samples": int(samples["ml_ready"].sum()),
        "ml_ready_rate": round(float(samples["ml_ready"].mean()), 4),
        "not_ready_reasons": samples["not_ready_reason"].value_counts().to_dict(),
        "scenes_per_split": samples.groupby("split")["scene_name"].nunique().to_dict(),
        "samples_per_split": samples[samples["ml_ready"]].groupby("split").size().to_dict(),
        "regression_set_samples": int((samples["in_regression_set"] & samples["ml_ready"]).sum()),
        "boxes_2d": len(labels),
        "boxes_per_class": labels["det_class"].value_counts().sort_index().to_dict(),
        "boxes_dropped": dropped,
    }
    write_json(summary, paths.REPORTS / "curation.json")
    log.info("curation: %s", summary)
    return summary
