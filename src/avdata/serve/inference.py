"""ONNX Runtime detector: letterbox -> inference -> decode -> NMS.

Pure NumPy pre/post-processing, so the same class runs in the API container,
in the benchmark script and on a Jetson (with the TensorRT or CUDA execution
provider when available).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PREFERRED_PROVIDERS = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


@dataclass
class Detection:
    cls: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels

    def as_dict(self) -> dict:
        return {
            "class": self.cls,
            "confidence": round(self.confidence, 4),
            "box": [round(v, 1) for v in self.box],
        }


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio and pad to size x size (grey 114, like YOLO training)."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top : top + nh, left : left + nw] = resized
    return out, scale, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ovr <= iou]
    return keep


class Detector:
    def __init__(
        self,
        model_path: Path | str,
        card_path: Path | str | None = None,
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort

        available = ort.get_available_providers()
        chosen = [p for p in (providers or PREFERRED_PROVIDERS) if p in available]
        self.session = ort.InferenceSession(str(model_path), providers=chosen)
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = int(self.session.get_inputs()[0].shape[2])
        card = (
            json.loads(Path(card_path).read_text())
            if card_path and Path(card_path).exists()
            else {}
        )
        self.card = card
        self.classes: list[str] = card.get("classes") or [
            str(i) for i in range(self.session.get_outputs()[0].shape[1] - 4)
        ]
        self.version = card.get("model_version", "unknown")
        self.providers = self.session.get_providers()

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        lb, scale, pad = letterbox(img, self.imgsz)
        x = lb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return x, scale, pad

    def postprocess(
        self,
        out: np.ndarray,
        scale: float,
        pad: tuple[float, float],
        shape: tuple[int, int],
        conf: float,
        iou: float,
    ) -> list[Detection]:
        pred = out[0].T  # (N, 4 + nc): cx, cy, w, h, class scores
        scores = pred[:, 4:]
        cls = scores.argmax(1)
        best = scores[np.arange(len(scores)), cls]
        m = best >= conf
        if not m.any():
            return []
        pred, cls, best = pred[m], cls[m], best[m]
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])
        dets = []
        for c in np.unique(cls):  # class-aware NMS
            idx = np.where(cls == c)[0]
            for k in nms(boxes[idx], best[idx], iou):
                i = idx[k]
                name = self.classes[c] if c < len(self.classes) else str(c)
                dets.append(Detection(name, float(best[i]), tuple(map(float, boxes[i]))))
        return sorted(dets, key=lambda d: -d.confidence)

    def __call__(
        self, img: np.ndarray, conf: float = 0.25, iou: float = 0.5
    ) -> tuple[list[Detection], dict]:
        t0 = time.perf_counter()
        x, scale, pad = self.preprocess(img)
        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: x})[0]
        t2 = time.perf_counter()
        dets = self.postprocess(out, scale, pad, img.shape[:2], conf, iou)
        t3 = time.perf_counter()
        timing = {
            "preprocess_ms": (t1 - t0) * 1e3,
            "inference_ms": (t2 - t1) * 1e3,
            "postprocess_ms": (t3 - t2) * 1e3,
            "total_ms": (t3 - t0) * 1e3,
        }
        return dets, timing
