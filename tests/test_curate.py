import pandas as pd
import yaml

from avdata import paths
from avdata.curate.run import SplitLeakageError, check_no_leakage


def test_no_scene_in_two_splits(in_clean):
    samples = pd.read_parquet(paths.SAMPLES)
    check_no_leakage(samples)
    assert set(samples["split"]) == {"train", "val", "test"}


def test_leakage_is_detected():
    df = pd.DataFrame({"scene_name": ["a", "a"], "split": ["train", "test"]})
    try:
        check_no_leakage(df)
    except SplitLeakageError:
        return
    raise AssertionError("leakage not detected")


def test_scenario_tags(in_clean):
    samples = pd.read_parquet(paths.SAMPLES).drop_duplicates("scene_name").set_index("scene_name")
    assert samples.loc["scene-9002", "time_of_day"] == "night"
    assert samples.loc["scene-9003", "weather"] == "rain"
    assert samples.loc["scene-9003", "speed_bin"] == "stopped"
    assert samples.loc["scene-9001", "has_bicycle"]
    assert samples.loc["scene-9002", "crowded"]


def test_regression_set_is_hard_test_cases(in_clean):
    samples = pd.read_parquet(paths.SAMPLES)
    reg = samples[samples["in_regression_set"]]
    assert (reg["split"] == "test").all()
    assert (
        reg["tags"]
        .apply(lambda t: bool({"night", "rain", "has_bicycle", "crowded"} & set(t)))
        .all()
    )


def test_labels_inside_image(in_clean):
    labels = pd.read_parquet(paths.LABELS_2D)
    assert len(labels) > 0
    assert (labels["x1"] >= 0).all() and (labels["x2"] <= labels["img_w"]).all()
    assert (labels["y1"] >= 0).all() and (labels["y2"] <= labels["img_h"]).all()
    assert (labels["x2"] > labels["x1"]).all()
    assert "animal" not in set(labels["det_class"])  # unmapped categories are ignored


def test_yolo_export(in_clean):
    _, p = in_clean
    root = paths.YOLO_DIR
    cfg = yaml.safe_load((root / "data.yaml").read_text())
    assert list(cfg["names"].values()) == p.curation.class_names
    for split in ("train", "val", "test"):
        imgs = sorted(f.stem for f in (root / "images" / split).glob("*.jpg"))
        lbls = sorted(f.stem for f in (root / "labels" / split).glob("*.txt"))
        assert imgs == lbls and imgs
    line = next(
        ln
        for f in (root / "labels" / "train").glob("*.txt")
        for ln in f.read_text().split("\n")
        if ln
    )
    cls, *xywh = line.split()
    assert 0 <= int(cls) < len(p.curation.class_names)
    assert all(0 <= float(v) <= 1 for v in xywh)
