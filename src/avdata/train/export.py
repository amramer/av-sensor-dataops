"""EXPORT stage: PyTorch -> ONNX, plus a model card for traceability.

ONNX is the hand-off format for deployment: ONNX Runtime serves it in the
FastAPI service (CPU or CUDA), and on NVIDIA Jetson the same file is compiled
to a TensorRT engine with `trtexec` (see docs/jetson.md).
"""

from __future__ import annotations

import hashlib
import json
import shutil

from avdata import paths
from avdata.config import Params
from avdata.train.train import lineage_tags
from avdata.utils import get_logger, write_json

log = get_logger(__name__)


def run(p: Params) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(paths.MODELS / "detector.pt"))
    onnx_path = model.export(
        format="onnx", imgsz=p.train.imgsz, opset=17, simplify=True, dynamic=False, verbose=False
    )
    target = paths.MODELS / "detector.onnx"
    if str(onnx_path) != str(target):
        shutil.move(onnx_path, target)

    ev_file = paths.METRICS / "eval.json"
    ev = json.loads(ev_file.read_text()) if ev_file.exists() else {}
    card = {
        "name": "av-camera-detector",
        "model_version": hashlib.sha256(target.read_bytes()).hexdigest()[:12],
        "format": "onnx",
        "opset": 17,
        "input": {
            "shape": [1, 3, p.train.imgsz, p.train.imgsz],
            "layout": "NCHW",
            "dtype": "float32",
            "normalisation": "RGB / 255, letterboxed",
        },
        "classes": p.curation.class_names,
        "camera": p.curation.camera,
        "train_params": p.train.model_dump(),
        "eval": {
            "overall": ev.get("overall"),
            "regression_set": ev.get("slices", {}).get("regression_set"),
        },
        **lineage_tags(),
    }
    write_json(card, paths.MODELS / "model_card.json")
    log.info("exported %s (version %s)", target, card["model_version"])
    return card
