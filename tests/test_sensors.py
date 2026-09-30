import numpy as np
import pytest
from PIL import Image

from avdata import sensors
from avdata.fixture import _write_radar


def test_camera_features(tmp_path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (160, 90), (100, 100, 100)).save(p)
    f = sensors.camera_features(p, {"width": 160, "height": 90, "image_feature_width": 80})
    assert f["brightness"] == pytest.approx(100, abs=2)
    assert np.isnan(f["n_points"])


def test_camera_wrong_size_is_corrupt(tmp_path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (160, 90)).save(p)
    with pytest.raises(sensors.CorruptFileError):
        sensors.camera_features(p, {"width": 1600, "height": 900})


def test_truncated_lidar_is_corrupt(tmp_path):
    p = tmp_path / "x.pcd.bin"
    p.write_bytes(np.zeros((10, 5), np.float32).tobytes()[:-3])
    with pytest.raises(sensors.CorruptFileError):
        sensors.lidar_features(p, {})


def test_lidar_features(tmp_path):
    p = tmp_path / "x.pcd.bin"
    pts = np.zeros((4, 5), np.float32)
    pts[:, 0] = [3, 4, 0, 0]
    pts[:, 1] = [4, 3, 5, 0]
    p.write_bytes(pts.tobytes())
    f = sensors.lidar_features(p, {})
    assert f["n_points"] == 4 and f["max_range"] == pytest.approx(5)


def test_radar_pcd(tmp_path):
    p = tmp_path / "r.pcd"
    _write_radar(p, 7)
    assert sensors.radar_features(p, {})["n_points"] == 7
    p.write_bytes(p.read_bytes()[:-5])
    with pytest.raises(sensors.CorruptFileError):
        sensors.radar_features(p, {})


def test_unknown_modality_has_helpful_error():
    with pytest.raises(KeyError, match="REGISTRY"):
        sensors.get_modality("thermal")
