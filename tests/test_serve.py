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
    from fastapi.testclient import TestClient

    from avdata.serve import app as app_module

    with TestClient(app_module.app) as c:
        yield c


def _jpeg(w=128, h=128):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 90, 90)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "test123"}


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
