"""Rule-based quality checks on sensor data and labels.

Each check returns issue records with a severity:

  ERROR  the data is unusable or the pipeline contract is broken -> stage fails
  WARN   the data is usable but degraded -> tracked as a KPI, frame may be excluded

Adding a check = writing one function that returns a list of issues and
adding it to CHECKS.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from avdata.config import Params

ERROR, WARN = "ERROR", "WARN"


@dataclass
class Tables:
    frames: pd.DataFrame
    features: pd.DataFrame
    annotations: pd.DataFrame
    sync: pd.DataFrame
    projection: pd.DataFrame | None = None  # LiDAR-in-camera share per keyframe pair

    @property
    def frames_features(self) -> pd.DataFrame:
        return self.frames.merge(self.features, on="frame_token", how="left")


def _issues(
    df: pd.DataFrame,
    check: str,
    severity: str,
    entity_type: str,
    id_col: str,
    message: Callable[[pd.Series], str],
    value_col: str | None = None,
) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        out.append(
            {
                "check": check,
                "severity": severity,
                "entity_type": entity_type,
                "entity_id": r[id_col],
                "scene_name": r.get("scene_name"),
                "channel": r.get("channel"),
                "value": float(r[value_col]) if value_col and pd.notna(r[value_col]) else None,
                "message": message(r),
            }
        )
    return out


# ---------------------------------------------------------------- completeness / integrity
def missing_files(t: Tables, p: Params) -> list[dict]:
    bad = t.frames[~t.frames["file_exists"]]
    return _issues(
        bad,
        "missing_file",
        ERROR,
        "frame",
        "frame_token",
        lambda r: f"{r['filename']} referenced in metadata but not on disk",
    )


def corrupt_files(t: Tables, p: Params) -> list[dict]:
    ff = t.frames_features
    bad = ff[ff["file_exists"] & ~ff["decode_ok"].fillna(False).astype(bool)]
    return _issues(
        bad,
        "corrupt_file",
        ERROR,
        "frame",
        "frame_token",
        lambda r: f"{r['filename']}: {r['decode_error']}",
    )


def duplicate_frames(t: Tables, p: Params) -> list[dict]:
    dup = t.frames[t.frames.duplicated(["scene_name", "channel", "timestamp"], keep="first")]
    return _issues(
        dup,
        "duplicate_frame",
        ERROR,
        "frame",
        "frame_token",
        lambda r: f"second frame at t={r['timestamp']} on {r['channel']}",
    )


# ---------------------------------------------------------------- timing
def sync_offsets(t: Tables, p: Params) -> list[dict]:
    s = t.sync
    tol = p.quality.sync_tolerance_ms
    missing = s[~s["has_keyframe"]]
    late = s[s["has_keyframe"] & (s["offset_ms"].abs() > tol)]
    return _issues(
        missing,
        "missing_keyframe",
        WARN,
        "sample",
        "sample_token",
        lambda r: f"no {r['channel']} keyframe for this sample",
    ) + _issues(
        late,
        "sync_offset",
        WARN,
        "sample",
        "sample_token",
        lambda r: (
            f"{r['channel']} is {r['offset_ms']:+.1f} ms from "
            f"{p.sensors.reference_channel} (tolerance {tol} ms)"
        ),
        "offset_ms",
    )


def frame_gaps(t: Tables, p: Params) -> list[dict]:
    f = t.frames.copy()
    period_ms = 1000.0 / f["modality"].map(p.sensors.nominal_hz)
    f["expected_ms"] = period_ms
    gaps = f[f["dt_prev_ms"] > p.quality.gap_factor * period_ms]
    return _issues(
        gaps,
        "frame_gap",
        WARN,
        "frame",
        "frame_token",
        lambda r: (
            f"{r['dt_prev_ms']:.0f} ms since previous {r['channel']} frame "
            f"(nominal {r['expected_ms']:.0f} ms): ~"
            f"{round(r['dt_prev_ms'] / r['expected_ms']) - 1} frames dropped"
        ),
        "dt_prev_ms",
    )


# ---------------------------------------------------------------- sensor content
def low_lidar_points(t: Tables, p: Params) -> list[dict]:
    ff = t.frames_features
    bad = ff[
        (ff["modality"] == "lidar")
        & ff["decode_ok"].fillna(False).astype(bool)
        & (ff["n_points"] < p.quality.min_lidar_points)
    ]
    return _issues(
        bad,
        "low_lidar_points",
        WARN,
        "frame",
        "frame_token",
        lambda r: f"{int(r['n_points'])} points (< {p.quality.min_lidar_points})",
        "n_points",
    )


def night_metadata_mismatch(t: Tables, p: Params) -> list[dict]:
    """Scene says 'night' but the images are bright: the metadata may be wrong."""
    ff = t.frames_features
    cam = ff[(ff["modality"] == "camera") & ff["is_key_frame"]]
    night = cam["scene_description"].str.contains("night", case=False, na=False)
    bad = cam[night & (cam["brightness"] > p.quality.night_brightness_max)]
    return _issues(
        bad,
        "night_metadata_mismatch",
        WARN,
        "frame",
        "frame_token",
        lambda r: f"scene tagged night but brightness {r['brightness']:.0f}",
        "brightness",
    )


# ---------------------------------------------------------------- labels
def weak_labels(t: Tables, p: Params) -> list[dict]:
    a = t.annotations
    bad = a[a["det_class"].notna() & (a["num_lidar_pts"] < p.quality.min_box_lidar_points)]
    return _issues(
        bad,
        "weak_label",
        WARN,
        "annotation",
        "annotation_token",
        lambda r: f"{r['category']} box with {r['num_lidar_pts']} lidar points",
        "num_lidar_pts",
    )


CHECKS: list[Callable[[Tables, Params], list[dict]]] = [
    missing_files,
    corrupt_files,
    duplicate_frames,
    sync_offsets,
    frame_gaps,
    low_lidar_points,
    night_metadata_mismatch,
    weak_labels,
]


# ---------------------------------------------------------------- multi-sensor calibration
def projection_shares(t: Tables, p: Params) -> pd.DataFrame:
    """Share of each LiDAR keyframe's ground points that land inside the paired camera image.

    Ground returns surround the car evenly, so for a fixed camera and LiDAR the
    share mostly depends on the camera's field of view and mounting, not on
    scene content (objects are excluded on purpose: a truck next to the car
    would change the share). A frame far from the median points to a
    calibration problem: wrong intrinsics, wrong extrinsics, or data logged
    under the wrong sensor.
    """
    from avdata.fusion.frames import frame_from_rows, project
    from avdata.fusion.lift import estimate_ground

    cols = ["sample_token", "frame_token", "scene_name", "share", "n_points"]
    cam_ch, lid_ch = p.curation.camera, p.fusion.lidar_channel
    if not {cam_ch, lid_ch} <= set(p.sensors.enabled):
        return pd.DataFrame(columns=cols)
    ff = t.frames_features
    ok = ff["is_key_frame"] & ff["decode_ok"].fillna(False).astype(bool)
    cams = ff[ok & (ff["channel"] == cam_ch)]
    lids = ff[ok & (ff["channel"] == lid_ch)].set_index("sample_token")
    rows = []
    for c in cams.itertuples():
        if c.sample_token not in lids.index:
            continue
        lr = lids.loc[c.sample_token]
        fr = frame_from_rows(c, lr, p.data.raw_dir)
        ego = fr.points_ego
        ground = ego[2] < estimate_ground(ego) + 0.2
        _, _, inside = project(fr.points_cam[:, ground], fr.K, fr.width, fr.height)
        if inside.size == 0:
            continue
        rows.append(
            {
                "sample_token": c.sample_token,
                "frame_token": c.frame_token,
                "scene_name": c.scene_name,
                "share": float(inside.mean()),
                "n_points": int(inside.size),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def calibration_suspect(t: Tables, p: Params) -> list[dict]:
    proj = t.projection
    if proj is None or len(proj) < 3:
        return []
    med = float(proj["share"].median())
    tol = p.quality.projection_ratio_tolerance
    proj = proj.assign(ratio=proj["share"] / max(med, 1e-9))
    bad = proj[(proj["ratio"] > tol) | (proj["ratio"] < 1 / tol)]
    return _issues(
        bad,
        "calibration_suspect",
        WARN,
        "frame",
        "frame_token",
        lambda r: (
            f"{r['share']:.1%} of LiDAR ground points land in the image vs "
            f"median {med:.1%} (x{r['ratio']:.2f}): check intrinsics/extrinsics"
        ),
        "share",
    )


CHECKS.insert(CHECKS.index(weak_labels), calibration_suspect)


def frame_status(t: Tables, issues: pd.DataFrame) -> pd.DataFrame:
    """Per-frame QC verdict used downstream to decide ML readiness."""
    frames = t.frames[
        ["frame_token", "sample_token", "scene_name", "channel", "modality", "is_key_frame"]
    ].copy()
    fi = issues[issues["entity_type"] == "frame"]
    errors = set(fi.loc[fi["severity"] == ERROR, "entity_id"])
    warn_codes = (
        fi[fi["severity"] == WARN].groupby("entity_id")["check"].agg(lambda s: sorted(set(s)))
    )
    frames["qc_error"] = frames["frame_token"].isin(errors)
    frames["qc_warnings"] = (
        frames["frame_token"].map(warn_codes).apply(lambda v: v if isinstance(v, list) else [])
    )
    # a sample is out of sync for a channel -> that channel's keyframe is not ML-ready
    si = issues[(issues["entity_type"] == "sample")]
    bad_pairs = set(zip(si["entity_id"], si["channel"], strict=True))
    frames["sync_ok"] = [
        (s, c) not in bad_pairs or not k
        for s, c, k in zip(
            frames["sample_token"], frames["channel"], frames["is_key_frame"], strict=True
        )
    ]
    calib_bad = set(fi.loc[fi["check"] == "calibration_suspect", "entity_id"])
    frames["calib_ok"] = ~frames["frame_token"].isin(calib_bad)
    frames["qc_pass"] = ~frames["qc_error"] & frames["sync_ok"] & frames["calib_ok"]
    return frames


def label_status(t: Tables, issues: pd.DataFrame, p: Params) -> pd.DataFrame:
    a = t.annotations[
        [
            "annotation_token",
            "sample_token",
            "scene_name",
            "det_class",
            "visibility",
            "num_lidar_pts",
        ]
    ].copy()
    weak = set(issues.loc[issues["check"] == "weak_label", "entity_id"])
    a["label_quality"] = np.select(
        [a["annotation_token"].isin(weak), a["visibility"] < p.curation.min_visibility],
        ["weak", "low_visibility"],
        default="ok",
    )
    return a
