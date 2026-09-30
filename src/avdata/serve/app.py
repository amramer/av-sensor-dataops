"""FastAPI inference service: camera detection and camera + LiDAR 3D detection.

  GET  /                          interactive demo page
  GET  /health                    liveness + model version (Docker/K8s probes)
  GET  /model                     model card + fusion config
  POST /predict                   image -> 2D detections (JSON)
  POST /predict/image             image -> the same image with boxes drawn (JPEG)
  POST /predict3d                 image + LiDAR sweep + calibration -> 3D boxes (JSON)
  GET  /api/samples               bundled demo samples
  GET  /api/samples/{id}/detect3d full camera + LiDAR inference on a demo sample
  GET  /api/samples/{id}/render.jpg  camera (LiDAR depth, 3D boxes) + bird's-eye view
  GET  /metrics/                  Prometheus metrics

Besides latency and throughput, the service exports simple input statistics
(image brightness) and the predicted class mix. Shifts in these are an early
signal of data drift, e.g. the fleet driving more at night than the training
data covered.

Run locally:  uvicorn avdata.serve.app:app --port 8000
"""

from __future__ import annotations

import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageDraw, UnidentifiedImageError
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from avdata.fusion import metrics
from avdata.fusion.lift import Det2D, LiftParams, lift
from avdata.fusion.priors import as_sizes
from avdata.serve import demo
from avdata.serve.inference import Detector

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 15 * 1024 * 1024))
STATIC = Path(__file__).parent / "static"

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
FUSED = Counter("avdata_fused_boxes_total", "3D boxes by source", ["source"])
MODEL_INFO = Gauge("avdata_model_info", "Loaded model", ["version", "provider"])

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    det = Detector(
        Path(os.environ.get("MODEL_PATH", "models/detector.onnx")),
        Path(os.environ.get("MODEL_CARD", "models/model_card.json")),
    )
    state["detector"] = det
    cfg_path = Path(os.environ.get("FUSION_CONFIG", "models/fusion.json"))
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    state["fusion_cfg"] = cfg
    state["sizes"] = as_sizes(cfg.get("priors"))
    state["lift"] = LiftParams(**cfg.get("lift", {}))
    state["demo"] = demo.load(Path(os.environ.get("DEMO_DIR", "demo")))
    MODEL_INFO.labels(det.version, det.providers[0]).set(1)
    yield
    state.clear()


app = FastAPI(title="AV camera + LiDAR detector", version="0.2.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


# ------------------------------------------------------------------ helpers
async def _read_image(image: UploadFile) -> np.ndarray:
    raw = await image.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        REQUESTS.labels("too_large").inc()
        raise HTTPException(413, "image too large")
    try:
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    except (UnidentifiedImageError, OSError):
        REQUESTS.labels("bad_image").inc()
        raise HTTPException(400, "could not decode image") from None


def _detect(img: np.ndarray, conf: float, iou: float):
    det: Detector = state["detector"]
    dets, timing = det(img, conf=conf, iou=iou)
    LATENCY.observe(timing["total_ms"] / 1000)
    BRIGHTNESS.observe(float(img.mean()))
    for d in dets:
        DETECTIONS.labels(d.cls).inc()
    REQUESTS.labels("ok").inc()
    return dets, timing


def _fuse(fr, dets) -> tuple[list, dict]:
    timings: dict = {}
    d2 = [Det2D(d.cls, d.confidence, d.box) for d in dets]
    boxes = lift(fr, d2, state["sizes"], state["lift"], timings)
    for b in boxes:
        FUSED.labels(b.source).inc()
    return boxes, timings


def _sample(sample_id: str) -> demo.DemoSample:
    s = state["demo"].get(sample_id)
    if s is None:
        raise HTTPException(404, f"unknown sample {sample_id}")
    return s


def _jpeg(img: Image.Image, quality: int = 85) -> Response:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return Response(buf.getvalue(), media_type="image/jpeg")


# ------------------------------------------------------------------ core endpoints
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    det: Detector | None = state.get("detector")
    return {
        "status": "ok" if det else "loading",
        "model_version": det.version if det else None,
        "demo_samples": len(state.get("demo", {})),
    }


@app.get("/model")
def model_card() -> dict:
    det: Detector = state["detector"]
    return det.card | {"providers": det.providers, "fusion": state.get("fusion_cfg", {})}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.5, ge=0.0, le=1.0),
) -> dict:
    img = await _read_image(image)
    dets, timing = _detect(img, conf, iou)
    return {
        "model_version": state["detector"].version,
        "image_size": [int(img.shape[1]), int(img.shape[0])],
        "timing_ms": {k: round(v, 2) for k, v in timing.items()},
        "detections": [d.as_dict() for d in dets],
    }


@app.post("/predict/image")
async def predict_image(
    image: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.5, ge=0.0, le=1.0),
) -> Response:
    from avdata.viz.render import PRED, font

    img = await _read_image(image)
    dets, timing = _detect(img, conf, iou)
    out = Image.fromarray(img)
    draw = ImageDraw.Draw(out)
    f = font(max(12, out.width // 90))
    for d in dets:
        draw.rectangle(d.box, outline=PRED, width=max(2, out.width // 500))
        label = f"{d.cls} {d.confidence:.2f}"
        x, y = d.box[0], max(0, d.box[1] - f.size - 6)
        draw.rectangle([x, y, x + draw.textlength(label, font=f) + 8, y + f.size + 6], fill=PRED)
        draw.text((x + 4, y + 2), label, font=f, fill=(255, 255, 255))
    resp = _jpeg(out)
    resp.headers["X-Model-Version"] = state["detector"].version
    resp.headers["X-Inference-Ms"] = f"{timing['total_ms']:.1f}"
    resp.headers["X-Detections"] = str(len(dets))
    return resp


@app.post("/predict3d")
async def predict3d(
    image: UploadFile = File(..., description="camera image"),
    lidar: UploadFile = File(..., description="float32 LiDAR points, N x lidar_dims, LiDAR frame"),
    calib: str = Form(
        ..., description="JSON: camera_intrinsic, width, height, camera, lidar poses"
    ),
    lidar_dims: int = Form(5),
    conf: float = Query(0.25, ge=0.0, le=1.0),
) -> dict:
    img = await _read_image(image)
    raw = await lidar.read()
    if lidar_dims < 3 or len(raw) % (4 * lidar_dims):
        raise HTTPException(400, f"LiDAR payload is not N x {lidar_dims} float32 values")
    try:
        pts = np.frombuffer(raw, np.float32).reshape(-1, lidar_dims)
        fr = demo.frame_from_calib(json.loads(calib), pts)
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(400, f"bad calibration: {e}") from None
    dets, timing = _detect(img, conf, 0.5)
    boxes, ft = _fuse(fr, dets)
    return {
        "model_version": state["detector"].version,
        "timing_ms": {"detector": round(timing["total_ms"], 2)}
        | {k: round(v, 2) for k, v in ft.items()},
        "boxes_ego": [b.as_dict() for b in boxes],
        "boxes_global": [b.to_global(fr).as_dict() for b in boxes],
    }


# ------------------------------------------------------------------ demo endpoints
@app.get("/api/samples")
def samples() -> list[dict]:
    keys = ("id", "scene", "description", "location", "tags", "n_gt", "t")
    return [{k: s.meta[k] for k in keys} for s in state["demo"].values()]


@app.get("/api/samples/{sample_id}/camera.jpg")
def sample_camera(sample_id: str) -> FileResponse:
    return FileResponse(_sample(sample_id).camera_path, media_type="image/jpeg")


def _run_sample(s: demo.DemoSample, conf: float):
    t0 = time.perf_counter()
    img = np.asarray(Image.open(s.camera_path).convert("RGB"))
    fr = s.frame()
    dets, timing = _detect(img, conf, 0.5)
    boxes, ft = _fuse(fr, dets)
    glob = [b.to_global(fr) for b in boxes]
    gt = [dict(g, sample=s.id) for g in s.meta["gt"]]
    pred = [
        {
            "sample": s.id,
            "cls": b.cls,
            "center": b.center,
            "size": b.size,
            "yaw": b.yaw,
            "score": b.score,
        }
        for b in glob
    ]
    matched = 0
    for cls in {g["cls"] for g in gt}:
        _, _, pairs = metrics.match(
            [p for p in pred if p["cls"] == cls], [g for g in gt if g["cls"] == cls], 2.0
        )
        matched += len(pairs)
    total = (time.perf_counter() - t0) * 1e3
    ego_xy = np.asarray(fr.cam.ego_t[:2])
    result = {
        "model_version": state["detector"].version,
        "timing_ms": {
            "detector": round(timing["total_ms"], 1),
            "fusion": round(sum(ft.values()), 1),
            "total": round(total, 1),
        },
        "gt_in_view": len(gt),
        "gt_matched_2m": matched,
        "boxes": [
            b.as_dict()
            | {"distance": round(float(np.hypot(*(np.asarray(b.center[:2]) - ego_xy))), 1)}
            for b in glob
        ],
    }
    return fr, img, glob, gt, result


@app.get("/api/samples/{sample_id}/detect3d")
def sample_detect3d(sample_id: str, conf: float = Query(0.25, ge=0.0, le=1.0)) -> dict:
    return _run_sample(_sample(sample_id), conf)[-1]


@app.get("/api/samples/{sample_id}/render.jpg")
def sample_render(
    sample_id: str,
    conf: float = Query(0.25, ge=0.0, le=1.0),
    lidar: bool = True,
    gt: bool = True,
) -> Response:
    from avdata.viz import render

    s = _sample(sample_id)
    fr, img, boxes, gts, res = _run_sample(s, conf)
    g = gts if gt else []
    cam = render.render_camera(Image.fromarray(img), fr, g, boxes, lidar=lidar)
    bev = render.render_bev(fr, g, boxes)
    title = f"{s.meta['scene']} · {s.meta['description'][:70]}"
    sub = (
        f"{len(boxes)} fused 3D boxes · {res['gt_matched_2m']}/{res['gt_in_view']} ground-truth "
        f"objects matched (2 m) · {res['timing_ms']['total']:.0f} ms · model {res['model_version']}"
    )
    return _jpeg(render.compose(cam, bev, title, sub, height=540), quality=82)
