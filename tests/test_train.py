"""Smoke test of train -> evaluate -> gate -> export on CPU (slow, needs ultralytics)."""

import json

import pytest

from avdata import paths
from avdata.config import load_params

pytestmark = pytest.mark.train
pytest.importorskip("ultralytics")


def test_train_evaluate_export(in_clean, monkeypatch):
    import yaml

    ws, _ = in_clean
    params = yaml.safe_load((ws / "params.yaml").read_text())
    params["train"].update(epochs=1, imgsz=160, batch=4, workers=0, device="cpu")
    (ws / "params_smoke.yaml").write_text(yaml.safe_dump(params))
    p = load_params(ws / "params_smoke.yaml")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{ws / 'mlflow.db'}")

    from avdata.train import evaluate, export, gate, train

    train.run(p)
    assert (paths.MODELS / "detector.pt").exists()
    # the DVC-tracked dataset must not be modified by training (no label caches)
    assert not list(paths.YOLO_DIR.rglob("*.cache"))

    ev = evaluate.run(p)
    assert "regression_set" in ev["slices"]
    assert {"map50", "map50_95", "precision", "recall"} <= set(ev["overall"])

    result = gate.run(p)
    assert isinstance(result["passed"], bool)

    card = export.run(p)
    assert (paths.MODELS / "detector.onnx").exists()
    assert card["classes"] == p.curation.class_names
    assert json.loads((paths.MODELS / "model_card.json").read_text())["model_version"]
