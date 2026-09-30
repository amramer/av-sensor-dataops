"""Camera + LiDAR fusion: metrics, lifting on the synthetic data, visual outputs, demo bundle."""

import json

import numpy as np
import pandas as pd
import pytest

from avdata import paths
from avdata.fusion import metrics
from avdata.fusion.lift import dominant_cluster


# ------------------------------------------------------------------ metrics
def _box(sample, cls, x, y, score=1.0, yaw=0.0):
    return {
        "sample": sample,
        "cls": cls,
        "center": [x, y, 0.8],
        "size": [1.9, 4.5, 1.6],
        "yaw": yaw,
        "score": score,
    }


def test_perfect_predictions_score_one():
    gts = [_box("s1", "car", 10, 0), _box("s1", "car", 20, 3), _box("s2", "pedestrian", 5, 1)]
    ev = metrics.evaluate([dict(g) for g in gts], gts)
    assert ev["mAP"] == pytest.approx(1.0)
    assert ev["mATE"] == pytest.approx(0.0)
    assert ev["NDS_lite"] == pytest.approx(1.0)


def test_centre_distance_thresholds():
    gts = [_box("s1", "car", 10, 0)]
    preds = [_box("s1", "car", 13, 0)]  # 3 m off: only the 4 m threshold matches
    ev = metrics.evaluate(preds, gts)
    assert ev["per_class"]["car"]["ap_by_threshold"] == {"0.5m": 0, "1m": 0, "2m": 0, "4m": 1.0}
    assert ev["mAP"] == pytest.approx(0.25)


def test_no_match_across_samples_or_classes():
    gts = [_box("s1", "car", 10, 0)]
    preds = [_box("s2", "car", 10, 0), _box("s1", "pedestrian", 10, 0)]
    assert metrics.evaluate(preds, gts)["mAP"] == 0.0


def test_orientation_error_is_modulo_pi():
    gts = [_box("s1", "car", 10, 0, yaw=0.1)]
    ev = metrics.evaluate([_box("s1", "car", 10, 0, yaw=0.1 + np.pi)], gts)
    assert ev["mAOE"] == pytest.approx(0.0, abs=1e-6)


def test_cluster_uses_depth_prior_to_skip_occluder():
    depths = np.r_[
        np.full(40, 15.0) + np.linspace(0, 2, 40), np.full(25, 21.0) + np.linspace(0, 3, 25)
    ]
    no_prior = dominant_cluster(depths, 0.8)
    with_prior = dominant_cluster(depths, 0.8, expected=21.0)
    assert depths[no_prior].min() < 16 and depths[with_prior].min() >= 21


# ------------------------------------------------------------------ pipeline on the fixture
@pytest.fixture(scope="module")
def fused(clean_ws):
    from avdata.fusion import run as fusion

    ws, p = clean_ws
    import os

    old = os.getcwd()
    os.chdir(ws)
    result = fusion.run(p)  # no detector model in this workspace -> oracle 2D only
    yield ws, p, result
    os.chdir(old)


def test_oracle_lifting_is_accurate(fused):
    _, _, r = fused
    o = r["oracle_2d"]
    assert r["test_samples"] > 0 and r["gt_boxes"] > 0
    assert o["mAP"] > 0.8
    assert o["mATE"] < 0.5
    assert o["lidar_share"] > 0.5
    assert "detector" not in r  # no model trained in this workspace


def test_fusion_outputs(fused):
    ws, _, _ = fused
    boxes = pd.read_parquet(ws / "data/predictions/boxes3d.parquet")
    assert set(boxes["source"]) <= {"lidar", "mono"} and len(boxes)
    cfg = json.loads((ws / "models/fusion.json").read_text())
    assert cfg["priors"]["car"]["source"] == "data"  # measured on the training split
    res = json.loads((ws / "reports/results_nusc.json").read_text())
    first = next(iter(res["results"].values()))[0]
    assert {
        "sample_token",
        "translation",
        "size",
        "rotation",
        "detection_name",
        "detection_score",
        "velocity",
        "attribute_name",
    } <= set(first)


def test_visual_outputs(fused):
    from avdata.viz import run as viz

    ws, p, _ = fused
    import os

    os.chdir(ws)
    summary = viz.run(p)
    assert summary["samples"] > 0 and summary["gifs"] >= 1
    out = ws / "reports/viz"
    assert (out / "index.html").read_text().count("<img") >= summary["samples"]
    from PIL import Image

    img = Image.open(next((out / "samples").glob("*.jpg")))
    assert img.width > 1500 and img.height > 600  # camera + bird's-eye view side by side


def test_demo_bundle(fused):
    from avdata.serve import demo

    ws, p, _ = fused
    import os

    os.chdir(ws)
    info = demo.build(p, n=3, out=ws / "demo")
    assert info["samples"] == 3
    samples = demo.load(ws / "demo")
    s = next(iter(samples.values()))
    fr = s.frame()
    assert fr.points_lidar.shape[1] == 4 and fr.K.shape == (3, 3)
    assert s.meta.get("gt")


def test_calibration_fault_is_caught(faulty_ws):
    from avdata.etl import extract, features, transform
    from avdata.quality import run as quality

    _, p = faulty_ws
    for stage in (extract, transform, features):
        stage.run(p)
    with pytest.raises(quality.QualityGateError):
        quality.run(p)
    issues = pd.read_parquet(paths.QUALITY_REPORTS / "issues.parquet")
    calib = issues[issues["check"] == "calibration_suspect"]
    assert len(calib) and set(calib["scene_name"]) == {"scene-9003"}
    qc = pd.read_parquet(paths.FRAMES_QC)
    assert not qc.loc[qc["frame_token"].isin(calib["entity_id"]), "qc_pass"].any()
