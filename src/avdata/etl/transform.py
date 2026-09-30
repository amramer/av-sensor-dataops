"""TRANSFORM: bronze tables -> silver frame, annotation and sync tables.

silver/frames.parquet       one row per sensor frame (keyframes + sweeps) of the
                            enabled channels, with calibration, ego pose, scene
                            context, file status and inter-frame timing
silver/annotations.parquet  one row per 3D box, with category, detection class,
                            visibility and attributes resolved
silver/sync.parquet         one row per (keyframe sample, channel) with the time
                            offset to the reference sensor
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from avdata import paths
from avdata.config import Params
from avdata.utils import get_logger, read_parquet, write_parquet

log = get_logger(__name__)


def _bronze(name: str) -> pd.DataFrame:
    return read_parquet(paths.BRONZE / f"{name}.parquet")


def _scene_context(p: Params) -> pd.DataFrame:
    scene = _bronze("scene").rename(
        columns={"token": "scene_token", "name": "scene_name", "description": "scene_description"}
    )
    logs = _bronze("log").rename(columns={"token": "log_token"})
    ctx = scene.merge(logs[["log_token", "location", "vehicle", "date_captured"]], on="log_token")
    if p.data.scenes != "all":
        unknown = set(p.data.scenes) - set(ctx["scene_name"])
        if unknown:
            raise ValueError(f"scenes not in the source release: {sorted(unknown)}")
        ctx = ctx[ctx["scene_name"].isin(p.data.scenes)]
    return ctx[
        ["scene_token", "scene_name", "scene_description", "location", "vehicle", "date_captured"]
    ]


def build_frames(p: Params) -> pd.DataFrame:
    sd = _bronze("sample_data")
    cs = _bronze("calibrated_sensor").rename(
        columns={
            "token": "calibrated_sensor_token",
            "translation": "sensor_translation",
            "rotation": "sensor_rotation",
        }
    )
    sensor = _bronze("sensor").rename(columns={"token": "sensor_token"})
    ego = _bronze("ego_pose").rename(
        columns={
            "token": "ego_pose_token",
            "translation": "ego_translation",
            "rotation": "ego_rotation",
            "timestamp": "ego_timestamp",
        }
    )
    sample = _bronze("sample").rename(
        columns={"token": "sample_token", "timestamp": "sample_timestamp"}
    )

    df = (
        sd.merge(
            cs[
                [
                    "calibrated_sensor_token",
                    "sensor_token",
                    "sensor_translation",
                    "sensor_rotation",
                    "camera_intrinsic",
                ]
            ],
            on="calibrated_sensor_token",
        )
        .merge(sensor[["sensor_token", "channel", "modality"]], on="sensor_token")
        .merge(ego[["ego_pose_token", "ego_translation", "ego_rotation"]], on="ego_pose_token")
        .merge(sample[["sample_token", "sample_timestamp", "scene_token"]], on="sample_token")
        .merge(_scene_context(p), on="scene_token")
    )
    df = df[df["channel"].isin(p.sensors.enabled)]
    missing = set(p.sensors.enabled) - set(df["channel"])
    if missing:
        raise ValueError(f"enabled channels not present in the data: {sorted(missing)}")
    if not p.etl.include_sweeps:
        df = df[df["is_key_frame"]]

    df = df.rename(columns={"token": "frame_token"})
    df["timestamp"] = df["timestamp"].astype("int64")
    df["is_key_frame"] = df["is_key_frame"].astype(bool)
    ego_xyz = np.vstack(df["ego_translation"].to_numpy())
    df["ego_x"], df["ego_y"], df["ego_z"] = ego_xyz[:, 0], ego_xyz[:, 1], ego_xyz[:, 2]

    # file status (existence and size) - the cheapest completeness check
    raw = p.data.raw_dir
    sizes = [(raw / f).stat().st_size if (raw / f).exists() else -1 for f in df["filename"]]
    df["file_size_bytes"] = np.asarray(sizes, dtype="int64")
    df["file_exists"] = df["file_size_bytes"] >= 0

    # inter-frame timing per channel and scene (for dropped-frame detection)
    df = df.sort_values(["scene_name", "channel", "timestamp"]).reset_index(drop=True)
    df["dt_prev_ms"] = df.groupby(["scene_name", "channel"])["timestamp"].diff() / 1000.0

    # ego speed from consecutive poses of the same channel (m/s)
    dxy = df.groupby(["scene_name", "channel"])[["ego_x", "ego_y"]].diff()
    df["ego_speed_mps"] = np.hypot(dxy["ego_x"], dxy["ego_y"]) / (df["dt_prev_ms"] / 1000.0)

    cols = [
        "frame_token",
        "sample_token",
        "scene_token",
        "scene_name",
        "scene_description",
        "location",
        "vehicle",
        "date_captured",
        "channel",
        "modality",
        "timestamp",
        "sample_timestamp",
        "is_key_frame",
        "filename",
        "file_exists",
        "file_size_bytes",
        "width",
        "height",
        "dt_prev_ms",
        "ego_x",
        "ego_y",
        "ego_z",
        "ego_speed_mps",
        "ego_translation",
        "ego_rotation",
        "sensor_translation",
        "sensor_rotation",
        "camera_intrinsic",
    ]
    return df[cols]


def build_annotations(p: Params, frames: pd.DataFrame) -> pd.DataFrame:
    ann = _bronze("sample_annotation").rename(columns={"token": "annotation_token"})
    inst = _bronze("instance").rename(columns={"token": "instance_token"})
    cat = _bronze("category").rename(columns={"token": "category_token", "name": "category"})
    attr = _bronze("attribute").set_index("token")["name"].to_dict()

    df = ann.merge(inst[["instance_token", "category_token"]], on="instance_token").merge(
        cat[["category_token", "category"]], on="category_token"
    )
    samples = frames[["sample_token", "scene_name"]].drop_duplicates()
    df = df.merge(samples, on="sample_token")  # keeps only ingested scenes
    df["visibility"] = df["visibility_token"].astype(int)
    df["attributes"] = df["attribute_tokens"].apply(lambda toks: [attr.get(t, t) for t in toks])
    df["det_class"] = df["category"].map(p.curation.classes)  # NaN = not a detection class
    cols = [
        "annotation_token",
        "sample_token",
        "scene_name",
        "instance_token",
        "category",
        "det_class",
        "visibility",
        "attributes",
        "translation",
        "size",
        "rotation",
        "num_lidar_pts",
        "num_radar_pts",
    ]
    return (
        df[cols]
        .sort_values(["scene_name", "sample_token", "annotation_token"])
        .reset_index(drop=True)
    )


def build_sync(p: Params, frames: pd.DataFrame) -> pd.DataFrame:
    """Offset of each channel's keyframe to the reference channel's keyframe."""
    key = frames[frames["is_key_frame"]][["sample_token", "scene_name", "channel", "timestamp"]]
    ref = key[key["channel"] == p.sensors.reference_channel][["sample_token", "timestamp"]]
    ref = ref.rename(columns={"timestamp": "ref_timestamp"})
    samples = frames[["sample_token", "scene_name"]].drop_duplicates()
    grid = samples.merge(pd.DataFrame({"channel": p.sensors.enabled}), how="cross")
    sync = grid.merge(key, on=["sample_token", "scene_name", "channel"], how="left").merge(
        ref, on="sample_token", how="left"
    )
    sync["has_keyframe"] = sync["timestamp"].notna()
    sync["offset_ms"] = (sync["timestamp"] - sync["ref_timestamp"]) / 1000.0
    return sync.sort_values(["scene_name", "sample_token", "channel"]).reset_index(drop=True)


def run(p: Params) -> dict:
    frames = build_frames(p)
    annotations = build_annotations(p, frames)
    sync = build_sync(p, frames)
    write_parquet(frames, paths.FRAMES)
    write_parquet(annotations, paths.ANNOTATIONS)
    write_parquet(sync, paths.SYNC)
    summary = {
        "scenes": int(frames["scene_name"].nunique()),
        "samples": int(frames["sample_token"].nunique()),
        "frames": len(frames),
        "frames_per_channel": frames["channel"].value_counts().sort_index().to_dict(),
        "annotations": len(annotations),
    }
    log.info("silver built: %s", summary)
    return summary
