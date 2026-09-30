"""3D boxes -> 2D camera labels.

For every camera keyframe, each annotated 3D box of a detection class is
transformed global -> ego -> camera and projected with the intrinsics. Boxes
are dropped (and the reason counted) when they are behind the camera,
outside the image, too small, poorly visible, or marked as weak labels by the
quality stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from avdata import geometry
from avdata.config import Params

DISTANCE_BINS = [(0, 20, "near"), (20, 40, "mid"), (40, np.inf, "far")]


def distance_bin(d: float) -> str:
    return next(name for lo, hi, name in DISTANCE_BINS if lo <= d < hi)


def project_labels(
    cam_frames: pd.DataFrame, annotations: pd.DataFrame, labels_qc: pd.DataFrame, p: Params
) -> tuple[pd.DataFrame, dict[str, int]]:
    c = p.curation
    class_id = {name: i for i, name in enumerate(c.class_names)}
    ann = annotations[annotations["det_class"].notna()].merge(
        labels_qc[["annotation_token", "label_quality"]], on="annotation_token", how="left"
    )
    by_sample = {k: g for k, g in ann.groupby("sample_token")}
    dropped = {"behind_or_outside": 0, "too_small": 0, "low_visibility": 0, "weak_label": 0}
    rows = []
    for f in cam_frames.itertuples():
        boxes = by_sample.get(f.sample_token)
        if boxes is None:
            continue
        K = np.array([list(r) for r in f.camera_intrinsic], dtype=np.float64)
        for b in boxes.itertuples():
            if b.label_quality == "weak":
                dropped["weak_label"] += 1
                continue
            if b.visibility < c.min_visibility:
                dropped["low_visibility"] += 1
                continue
            corners = geometry.box_corners(
                np.array(b.translation), np.array(b.size), np.array(b.rotation)
            )
            cam = geometry.global_to_sensor(
                corners, f.ego_translation, f.ego_rotation, f.sensor_translation, f.sensor_rotation
            )
            box = geometry.box_to_2d(cam, K, int(f.width), int(f.height))
            if box is None:
                dropped["behind_or_outside"] += 1
                continue
            x1, y1, x2, y2 = box
            if min(x2 - x1, y2 - y1) < c.min_box_px:
                dropped["too_small"] += 1
                continue
            dist = float(
                np.hypot(
                    b.translation[0] - f.ego_translation[0], b.translation[1] - f.ego_translation[1]
                )
            )
            rows.append(
                {
                    "sample_token": f.sample_token,
                    "frame_token": f.frame_token,
                    "annotation_token": b.annotation_token,
                    "det_class": b.det_class,
                    "class_id": class_id[b.det_class],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "img_w": int(f.width),
                    "img_h": int(f.height),
                    "distance_m": dist,
                    "distance_bin": distance_bin(dist),
                    "visibility": int(b.visibility),
                    "num_lidar_pts": int(b.num_lidar_pts),
                }
            )
    cols = [
        "sample_token",
        "frame_token",
        "annotation_token",
        "det_class",
        "class_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "img_w",
        "img_h",
        "distance_m",
        "distance_bin",
        "visibility",
        "num_lidar_pts",
    ]
    return pd.DataFrame(rows, columns=cols), dropped
