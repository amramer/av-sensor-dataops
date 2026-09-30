"""Latency benchmark for the exported ONNX model on the current machine.

    avdata benchmark --runs 200
    avdata benchmark --providers CPUExecutionProvider

The same command runs on a laptop, a cloud GPU or an NVIDIA Jetson (with
onnxruntime-gpu / the TensorRT execution provider installed), and writes one
JSON per device into reports/benchmarks/, so results from different hardware
sit side by side. See docs/jetson.md for the Jetson procedure.
"""

from __future__ import annotations

import platform
import socket
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from avdata import paths
from avdata.serve.inference import Detector
from avdata.utils import get_logger, utc_now, write_json

log = get_logger(__name__)


def _images(n: int) -> list[np.ndarray]:
    files = sorted((paths.YOLO_DIR / "images" / "test").glob("*.jpg"))[:n]
    if files:
        return [np.asarray(Image.open(f).convert("RGB")) for f in files]
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (900, 1600, 3), dtype=np.uint8)]


def run(
    runs: int = 100,
    warmup: int = 10,
    providers: list[str] | None = None,
    model: Path = paths.MODELS / "detector.onnx",
) -> dict:
    det = Detector(model, paths.MODELS / "model_card.json", providers=providers)
    imgs = _images(20)
    for i in range(warmup):
        det(imgs[i % len(imgs)])
    stages: dict[str, list[float]] = {
        "preprocess_ms": [],
        "inference_ms": [],
        "postprocess_ms": [],
        "total_ms": [],
    }
    t0 = time.perf_counter()
    for i in range(runs):
        _, t = det(imgs[i % len(imgs)])
        for k in stages:
            stages[k].append(t[k])
    wall = time.perf_counter() - t0

    def pct(v: list[float], q: float) -> float:
        return round(float(np.percentile(v, q)), 2)

    result = {
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "provider": det.providers[0],
        "model_version": det.version,
        "imgsz": det.imgsz,
        "runs": runs,
        "fps": round(runs / wall, 2),
        "latency": {
            k: {
                "mean": round(statistics.fmean(v), 2),
                "p50": pct(v, 50),
                "p95": pct(v, 95),
                "p99": pct(v, 99),
            }
            for k, v in stages.items()
        },
        "measured_at": utc_now(),
    }
    out = paths.REPORTS / "benchmarks" / f"{result['host']}_{result['provider']}.json"
    write_json(result, out)
    log.info(
        "%s on %s: p50 %.1f ms, p95 %.1f ms, %.1f FPS",
        det.version,
        det.providers[0],
        result["latency"]["total_ms"]["p50"],
        result["latency"]["total_ms"]["p95"],
        result["fps"],
    )
    return result
