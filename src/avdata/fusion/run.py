"""FUSE3D stage: camera + LiDAR 3D detection on the test split, and its evaluation.

Two modes are run and reported side by side:

  oracle_2d  2D boxes come from the ground-truth labels. Measures only the
             3D lifting (geometry, calibration, time alignment, clustering).
  detector   2D boxes come from the trained camera detector (ONNX). Measures
             the full camera + LiDAR system as it would run in the vehicle.

Separating the two is standard practice: it tells you whether a weak 3D
result comes from the 2D detector or from the fusion step.

Outputs
  models/fusion.json          size priors (from training data) + fusion params
  data/predictions/boxes3d.parquet   every predicted box, both modes
  data/predictions/gt3d.parquet      evaluated ground truth (in range, in camera view)
  metrics/eval3d.json         mAP, ATE, ASE, AOE, NDS-lite, per class, by distance
  reports/results_nusc.json   detector boxes in the nuScenes submission format
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from avdata import geometry, paths
from avdata.config import Params
from avdata.curate.labels2d import distance_bin
from avdata.fusion import metrics, priors
from avdata.fusion.frames import FusionFrame, frame_from_rows, project, quat_yaw, yaw_quat
from avdata.fusion.lift import Det2D, LiftParams, lift
from avdata.utils import get_logger, read_parquet, write_json, write_parquet

log = get_logger(__name__)
PRED_DIR = paths.DATA / "predictions"


def lift_params(p: Params) -> LiftParams:
    f = p.fusion
    return LiftParams(
        box_shrink=f.box_shrink,
        ground_offset_m=f.ground_offset_m,
        min_points=f.min_points,
        cluster_gap_m=f.cluster_gap_m,
        near_percentile=f.near_percentile,
    )


def fusion_config(p: Params, prior_table: dict) -> dict:
    return {
        "priors": prior_table,
        "lift": lift_params(p).__dict__,
        "score_threshold": p.fusion.score_threshold,
        "class_range_m": p.fusion.class_range_m,
    }


def test_frames(p: Params) -> pd.DataFrame:
    """Camera + LiDAR keyframe rows for every ML-ready test sample."""
    frames = read_parquet(paths.FRAMES)
    qc = read_parquet(paths.FRAMES_QC)[["frame_token", "qc_error"]]
    samples = read_parquet(paths.SAMPLES)
    test = samples[(samples["split"] == "test") & samples["ml_ready"]]
    key = frames[frames["is_key_frame"]].merge(qc, on="frame_token")
    cam = key[key["channel"] == p.curation.camera]
    lid = key[(key["channel"] == p.fusion.lidar_channel) & ~key["qc_error"]]
    pairs = cam.merge(lid, on="sample_token", suffixes=("", "_lidar"))
    pairs = pairs[pairs["sample_token"].isin(test["sample_token"])]
    tags = test.set_index("sample_token")["tags"]
    pairs["tags"] = pairs["sample_token"].map(tags)
    return pairs.sort_values(["scene_name", "timestamp"]).reset_index(drop=True)


def lidar_row(r) -> pd.Series:
    """The LiDAR half of a merged camera/LiDAR row, with plain column names."""
    cols = ["filename", "sensor_translation", "sensor_rotation", "ego_translation", "ego_rotation"]
    return pd.Series({c: getattr(r, f"{c}_lidar") for c in cols})


def build_frame(r, raw_dir: Path) -> FusionFrame:
    return frame_from_rows(r, lidar_row(r), raw_dir)


def gt_boxes(anns: pd.DataFrame, fr: FusionFrame, sample: str, class_range: dict) -> list[dict]:
    """Ground truth that the front camera can see: in range, in view, >= 1 LiDAR point."""
    out = []
    ego_xy = np.asarray(fr.cam.ego_t[:2])
    for a in anns.itertuples():
        if not isinstance(a.det_class, str) or a.num_lidar_pts < 1:
            continue
        c = np.asarray(a.translation, dtype=np.float64)
        dist = float(np.hypot(*(c[:2] - ego_xy)))
        if dist > class_range.get(a.det_class, 50):
            continue
        cc = geometry.global_to_sensor(
            c.reshape(3, 1), fr.cam.ego_t, fr.cam.ego_q, fr.cam.sensor_t, fr.cam.sensor_q
        )
        _, _, inside = project(cc, fr.K, fr.width, fr.height)
        if not inside[0]:
            continue
        out.append(
            {
                "sample": sample,
                "cls": a.det_class,
                "center": c.tolist(),
                "size": list(a.size),
                "yaw": quat_yaw(a.rotation),
                "score": 1.0,
                "distance": dist,
                "distance_bin": distance_bin(dist),
                "annotation_token": a.annotation_token,
                "num_lidar_pts": int(a.num_lidar_pts),
            }
        )
    return out


def detector_dets(detector, image_path: Path, threshold: float) -> tuple[list[Det2D], float]:
    img = np.asarray(Image.open(image_path).convert("RGB"))
    dets, timing = detector(img, conf=threshold)
    return [Det2D(d.cls, d.confidence, d.box) for d in dets], timing["total_ms"]


def run(p: Params) -> dict:
    from avdata.serve.inference import Detector

    anns_all = read_parquet(paths.ANNOTATIONS)
    samples = read_parquet(paths.SAMPLES)
    train_samples = set(samples.loc[samples["split"] == "train", "sample_token"])
    prior_table = priors.from_annotations(anns_all[anns_all["sample_token"].isin(train_samples)])
    sizes = priors.as_sizes(prior_table)
    lp = lift_params(p)
    cfg = fusion_config(p, prior_table)
    write_json(cfg, paths.MODELS / "fusion.json")

    labels2d = read_parquet(paths.LABELS_2D)
    by_frame = {k: g for k, g in labels2d.groupby("frame_token")}
    anns_by_sample = {k: g for k, g in anns_all.groupby("sample_token")}
    pairs = test_frames(p)
    model = paths.MODELS / "detector.onnx"
    detector = Detector(model, paths.MODELS / "model_card.json") if model.exists() else None
    class_range = p.fusion.class_range_m

    preds: dict[str, list[dict]] = {"oracle_2d": [], "detector": []}
    gts: list[dict] = []
    rows: list[dict] = []
    t_lift, t_det = [], []
    for r in pairs.itertuples():
        fr = build_frame(r, p.data.raw_dir)
        gt = gt_boxes(
            anns_by_sample.get(r.sample_token, anns_all.iloc[:0]), fr, r.sample_token, class_range
        )
        gts += gt
        lab = by_frame.get(r.frame_token)
        modes = {
            "oracle_2d": []
            if lab is None
            else [Det2D(b.det_class, 1.0, (b.x1, b.y1, b.x2, b.y2)) for b in lab.itertuples()]
        }
        if detector is not None:
            dets, ms = detector_dets(
                detector, p.data.raw_dir / r.filename, p.fusion.score_threshold
            )
            t_det.append(ms)
            modes["detector"] = dets
        for mode, dets in modes.items():
            timings: dict = {}
            boxes = [b.to_global(fr) for b in lift(fr, dets, sizes, lp, timings)]
            t_lift.append(timings.get("project_ms", 0) + timings.get("lift_ms", 0))
            ego_xy = np.asarray(fr.cam.ego_t[:2])
            for b in boxes:
                dist = float(np.hypot(b.center[0] - ego_xy[0], b.center[1] - ego_xy[1]))
                if dist > class_range.get(b.cls, 50):
                    continue
                d = {
                    "sample": r.sample_token,
                    "cls": b.cls,
                    "center": b.center,
                    "size": b.size,
                    "yaw": b.yaw,
                    "score": b.score,
                }
                preds[mode].append(d)
                rows.append(
                    {
                        "mode": mode,
                        "sample_token": r.sample_token,
                        "frame_token": r.frame_token,
                        "scene_name": r.scene_name,
                        **b.as_dict(),
                        "distance": round(dist, 2),
                    }
                )

    th = tuple(p.fusion.eval_thresholds_m)
    result: dict = {
        "camera": p.curation.camera,
        "lidar": p.fusion.lidar_channel,
        "test_samples": len(pairs),
        "gt_boxes": len(gts),
        "priors_source": {k: v["source"] for k, v in prior_table.items()},
    }
    for mode, P in preds.items():
        if mode == "detector" and detector is None:
            continue
        ev = metrics.evaluate(P, gts, th, p.fusion.tp_threshold_m)
        ev["by_distance"] = metrics.recall_by(P, gts, "distance_bin", p.fusion.tp_threshold_m)
        ev["lidar_share"] = (
            round(float(np.mean([r["source"] == "lidar" for r in rows if r["mode"] == mode])), 4)
            if P
            else None
        )
        result[mode] = ev
    result["timing_ms"] = {
        "lift_per_frame": round(float(np.mean(t_lift)), 2) if t_lift else None,
        "detector_per_frame": round(float(np.mean(t_det)), 2) if t_det else None,
    }

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    box_cols = [
        "mode",
        "sample_token",
        "frame_token",
        "scene_name",
        "cls",
        "score",
        "center",
        "size",
        "yaw",
        "n_points",
        "source",
        "frame",
        "box2d",
        "distance",
    ]
    write_parquet(pd.DataFrame(rows, columns=box_cols), PRED_DIR / "boxes3d.parquet")
    write_parquet(pd.DataFrame(gts), PRED_DIR / "gt3d.parquet")
    write_json(result, paths.METRICS / "eval3d.json")
    write_json(
        nuscenes_results(
            preds["detector"] or preds["oracle_2d"],
            "detector" if preds["detector"] else "oracle_2d",
        ),
        paths.REPORTS / "results_nusc.json",
    )
    summary = {
        m: {k: result[m][k] for k in ("mAP", "mATE", "mASE", "mAOE", "NDS_lite")}
        for m in preds
        if m in result
    }
    log.info("3D fusion on %d test samples: %s", len(pairs), json.dumps(summary))
    return result


def nuscenes_results(preds: list[dict], mode: str) -> dict:
    """Predictions in the nuScenes detection submission format."""
    results: dict[str, list] = {}
    for d in preds:
        results.setdefault(d["sample"], []).append(
            {
                "sample_token": d["sample"],
                "translation": [round(v, 3) for v in d["center"]],
                "size": [round(v, 3) for v in d["size"]],
                "rotation": [round(v, 6) for v in yaw_quat(d["yaw"])],
                "velocity": [0.0, 0.0],
                "detection_name": d["cls"],
                "detection_score": round(d["score"], 4),
                "attribute_name": "",
            }
        )
    return {
        "meta": {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": mode == "detector",
            "source_2d": mode,
        },
        "results": results,
    }
