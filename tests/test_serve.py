"""Inference service tests with a tiny hand-built ONNX model (no torch needed)."""

import io
import json

import numpy as np
import pytest
from PIL import Image

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

CLASSES = ["car", "pedestrian"]


def make_model(path, imgsz=64):
    """Model with YOLOv8's output layout (1, 4 + nc, N) returning fixed predictions."""
    preds = np.zeros((1, 4 + len(CLASSES), 3), dtype=np.float32)
    preds[0, :4, 0] = [32, 32, 20, 10]  # car at the centre, score 0.9
    preds[0, 4, 0] = 0.9
    preds[0, :4, 1] = [33, 32, 20, 10]  # duplicate car, lower score -> removed by NMS
    preds[0, 4, 1] = 0.6
    preds[0, :4, 2] = [10, 10, 4, 8]  # pedestrian below threshold
    preds[0, 5, 2] = 0.1
    const = helper.make_node("Constant", [], ["preds"], value=numpy_helper.from_array(preds))
    zero = helper.make_node("ReduceSum", ["images"], ["s"], keepdims=0)
    mul = helper.make_node("Mul", ["s", "zero_c"], ["z"])
    add = helper.make_node("Add", ["preds", "z"], ["output0"])
    graph = helper.make_graph(
        [const, zero, mul, add],
        "fake_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, imgsz, imgsz])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + len(CLASSES), 3])],
        initializer=[numpy_helper.from_array(np.array(0, dtype=np.float32), "zero_c")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    make_model(tmp_path / "m.onnx")
    (tmp_path / "card.json").write_text(
        json.dumps({"model_version": "test123", "classes": CLASSES})
    )
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "m.onnx"))
    monkeypatch.setenv("MODEL_CARD", str(tmp_path / "card.json"))
    monkeypatch.setenv("DEMO_DIR", str(tmp_path / "no-demo"))
    from fastapi.testclient import TestClient

    from avdata.serve import app as app_module

    with TestClient(app_module.app) as c:
        yield c


def _jpeg(w=128, h=128):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 90, 90)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "model_version": "test123",
        "demo_samples": 0,
    }


def test_predict_decodes_and_applies_nms(client):
    r = client.post("/predict", files={"image": ("x.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert [d["class"] for d in body["detections"]] == ["car"]
    x1, y1, x2, y2 = body["detections"][0]["box"]
    # 64px model input, 128px image -> scale 2: centre (32,32) -> (64,64), w 20 -> 40
    assert (x1, y1, x2, y2) == pytest.approx((44, 54, 84, 74), abs=1)


def test_letterbox_non_square(client):
    r = client.post("/predict", files={"image": ("x.jpg", _jpeg(256, 128), "image/jpeg")})
    x1, y1, x2, y2 = r.json()["detections"][0]["box"]
    assert (x1 + x2) / 2 == pytest.approx(128, abs=1) and (y1 + y2) / 2 == pytest.approx(64, abs=1)


def test_bad_image_is_rejected(client):
    r = client.post("/predict", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_metrics_exposed(client):
    client.post("/predict", files={"image": ("x.jpg", _jpeg(), "image/jpeg")})
    text = client.get("/metrics/").text
    assert "avdata_inference_seconds" in text and 'avdata_detections_total{cls="car"}' in text


# ------------------------------------------------------------------ camera + LiDAR demo endpoints
@pytest.fixture()
def demo_client(tmp_path, monkeypatch, clean_ws):
    import os

    from fastapi.testclient import TestClient

    from avdata.serve import app as app_module
    from avdata.serve import demo

    ws, p = clean_ws
    old = os.getcwd()
    os.chdir(ws)
    demo.build(p, n=2, out=tmp_path / "demo")
    os.chdir(old)
    make_model(tmp_path / "m.onnx")
    (tmp_path / "card.json").write_text(json.dumps({"model_version": "t", "classes": CLASSES}))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "m.onnx"))
    monkeypatch.setenv("MODEL_CARD", str(tmp_path / "card.json"))
    monkeypatch.setenv("DEMO_DIR", str(tmp_path / "demo"))
    monkeypatch.setenv("FUSION_CONFIG", str(tmp_path / "none.json"))
    with TestClient(app_module.app) as c:
        yield c, tmp_path / "demo"


def test_demo_page_and_samples(demo_client):
    c, _ = demo_client
    assert "Camera + LiDAR" in c.get("/").text
    samples = c.get("/api/samples").json()
    assert len(samples) == 2 and {"id", "scene", "n_gt", "t"} <= set(samples[0])
    assert c.get("/health").json()["demo_samples"] == 2


def test_demo_detect3d_and_render(demo_client):
    c, _ = demo_client
    sid = c.get("/api/samples").json()[0]["id"]
    r = c.get(f"/api/samples/{sid}/detect3d").json()
    assert {"boxes", "gt_in_view", "gt_matched_2m", "timing_ms"} <= set(r)
    assert all(b["source"] in ("lidar", "mono") and len(b["center"]) == 3 for b in r["boxes"])
    img = c.get(f"/api/samples/{sid}/render.jpg")
    assert img.headers["content-type"] == "image/jpeg" and img.content[:2] == b"\xff\xd8"
    assert c.get("/api/samples/nope/detect3d").status_code == 404


def test_predict3d_with_uploaded_frame(demo_client):
    c, demo_dir = demo_client
    sid = c.get("/api/samples").json()[0]["id"]
    meta = json.loads((demo_dir / sid / "meta.json").read_text())
    files = {
        "image": ("cam.jpg", (demo_dir / sid / "camera.jpg").read_bytes(), "image/jpeg"),
        "lidar": (
            "lidar.bin",
            (demo_dir / sid / "lidar.bin").read_bytes(),
            "application/octet-stream",
        ),
    }
    r = c.post(
        "/predict3d", files=files, data={"calib": json.dumps(meta["calib"]), "lidar_dims": "4"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["boxes_ego"]) == len(body["boxes_global"]) == 1  # the fake model finds one car
    bad = c.post("/predict3d", files=files, data={"calib": "{}", "lidar_dims": "4"})
    assert bad.status_code == 400


def test_predict_image_returns_annotated_jpeg(demo_client):
    c, demo_dir = demo_client
    sid = c.get("/api/samples").json()[0]["id"]
    r = c.post(
        "/predict/image",
        files={"image": ("x.jpg", (demo_dir / sid / "camera.jpg").read_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200 and r.headers["x-detections"] == "1"
    assert r.content[:2] == b"\xff\xd8"
