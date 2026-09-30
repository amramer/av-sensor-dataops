"""FastAPI inference service.

  GET  /health    liveness + model version (used by Docker/K8s probes)
  GET  /model     model card: version, classes, git commit, dataset version, eval
  POST /predict   image upload -> detections
  GET  /metrics   Prometheus metrics (scraped by Prometheus, charted in Grafana)

Besides latency and throughput, the service exports simple input statistics
(image brightness) and the predicted class mix. Shifts in these are an early
signal of data drift, e.g. the fleet driving more at night than the training
data covered.

Run locally:  uvicorn avdata.serve.app:app --port 8000
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from avdata.serve.inference import Detector

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 15 * 1024 * 1024))

REQUESTS = Counter("avdata_requests_total", "Prediction requests", ["status"])
LATENCY = Histogram(
    "avdata_inference_seconds",
    "End-to-end model latency",
    buckets=(0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56),
)
DETECTIONS = Counter("avdata_detections_total", "Detections by class", ["cls"])
BRIGHTNESS = Histogram(
    "avdata_input_brightness",
    "Mean grey level of input images",
    buckets=(20, 40, 60, 80, 100, 120, 140, 160, 200, 255),
)
MODEL_INFO = Gauge("avdata_model_info", "Loaded model", ["version", "provider"])

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    det = Detector(
        Path(os.environ.get("MODEL_PATH", "models/detector.onnx")),
        Path(os.environ.get("MODEL_CARD", "models/model_card.json")),
    )
    state["detector"] = det
    MODEL_INFO.labels(det.version, det.providers[0]).set(1)
    yield
    state.clear()


app = FastAPI(title="AV camera detector", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health() -> dict:
    det: Detector | None = state.get("detector")
    return {"status": "ok" if det else "loading", "model_version": det.version if det else None}


@app.get("/model")
def model_card() -> dict:
    det: Detector = state["detector"]
    return det.card | {"providers": det.providers}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.5, ge=0.0, le=1.0),
) -> dict:
    raw = await image.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        REQUESTS.labels("too_large").inc()
        raise HTTPException(413, "image too large")
    try:
        img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    except (UnidentifiedImageError, OSError):
        REQUESTS.labels("bad_image").inc()
        raise HTTPException(400, "could not decode image") from None

    det: Detector = state["detector"]
    dets, timing = det(img, conf=conf, iou=iou)
    LATENCY.observe(timing["total_ms"] / 1000)
    BRIGHTNESS.observe(float(img.mean()))
    for d in dets:
        DETECTIONS.labels(d.cls).inc()
    REQUESTS.labels("ok").inc()
    return {
        "model_version": det.version,
        "image_size": [int(img.shape[1]), int(img.shape[0])],
        "timing_ms": {k: round(v, 2) for k, v in timing.items()},
        "detections": [d.as_dict() for d in dets],
    }
