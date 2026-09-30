"""End-to-end tests of the data stages on the synthetic dataset."""

import json

import pandas as pd
import pytest

from avdata import paths
from avdata.config import load_params
from avdata.etl import extract, features, transform
from avdata.quality import run as quality


def test_bronze_manifest(in_clean):
    manifest = json.loads((paths.BRONZE / "_manifest.json").read_text())
    assert manifest["tables"]["scene"]["rows"] == 3
    assert all(len(t["sha256"]) == 64 for t in manifest["tables"].values())


def test_frames_only_enabled_channels(in_clean):
    frames = pd.read_parquet(paths.FRAMES)
    assert set(frames["channel"]) == {"CAM_FRONT", "LIDAR_TOP"}
    assert frames["file_exists"].all()
    # nominal rates recovered from timestamps
    lidar_dt = frames.loc[frames["channel"] == "LIDAR_TOP", "dt_prev_ms"].median()
    assert lidar_dt == pytest.approx(50, abs=1)


def test_sync_table_covers_every_sample_and_channel(in_clean):
    sync = pd.read_parquet(paths.SYNC)
    frames = pd.read_parquet(paths.FRAMES)
    assert len(sync) == frames["sample_token"].nunique() * 2
    assert sync["has_keyframe"].all()
    assert sync["offset_ms"].abs().max() < 50


def test_clean_data_passes_quality(in_clean):
    summary = json.loads((paths.QUALITY_REPORTS / "summary.json").read_text())
    assert summary["errors"] == 0
    assert summary["frame_pass_rate"] == 1.0


def test_faults_are_detected(faulty_ws):
    _, p = faulty_ws
    for stage in (extract, transform, features):
        stage.run(p)
    with pytest.raises(quality.QualityGateError):
        quality.run(p)
    issues = pd.read_parquet(paths.QUALITY_REPORTS / "issues.parquet")
    found = set(issues["check"])
    assert {
        "corrupt_file",
        "missing_file",
        "sync_offset",
        "frame_gap",
        "weak_label",
        "calibration_suspect",
    } <= found
    frames_qc = pd.read_parquet(paths.FRAMES_QC)
    assert not frames_qc["qc_pass"].all()


def test_adding_a_sensor_is_config_only(tmp_path, monkeypatch):
    """Enabling RADAR_FRONT needs no code change: it flows through ETL and QC."""
    from tests.conftest import _workspace

    ws = _workspace(
        tmp_path, faults=False, sensors={"enabled": ["CAM_FRONT", "LIDAR_TOP", "RADAR_FRONT"]}
    )
    monkeypatch.chdir(ws)
    p = load_params(ws / "params.yaml")
    assert "RADAR_FRONT" in p.sensors.enabled
    for stage in (extract, transform, features):
        stage.run(p)
    summary = quality.run(p)
    assert "RADAR_FRONT" in summary["frame_pass_rate_by_channel"]
    feats = pd.read_parquet(paths.FEATURES).merge(pd.read_parquet(paths.FRAMES), on="frame_token")
    assert (feats.loc[feats["channel"] == "RADAR_FRONT", "n_points"] == 12).all()


def test_scene_filter_simulates_incremental_ingest(tmp_path, monkeypatch):
    from tests.conftest import _workspace

    ws = _workspace(tmp_path, faults=False, data={"scenes": ["scene-9001", "scene-9002"]})
    monkeypatch.chdir(ws)
    p = load_params(ws / "params.yaml")
    extract.run(p)
    assert transform.run(p)["scenes"] == 2
