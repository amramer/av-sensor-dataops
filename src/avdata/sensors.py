"""Sensor modality registry.

Every modality knows how to (1) validate one raw file and (2) compute cheap
per-frame features. The ETL and quality stages only talk to this registry,
so adding a new sensor type means adding one `Modality` here and enabling its
channel in params.yaml. No other stage needs to change.

Currently registered: camera (.jpg), lidar (.pcd.bin), radar (.pcd).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

FEATURE_COLUMNS = [
    "brightness",
    "contrast",
    "sharpness",
    "n_points",
    "mean_range",
    "max_range",
    "mean_intensity",
]


class CorruptFileError(Exception):
    """Raised when a sensor file exists but cannot be decoded."""


@dataclass(frozen=True)
class Modality:
    name: str
    file_suffix: str
    features: Callable[[Path, dict], dict[str, float]]


def _empty() -> dict[str, float]:
    return {c: math.nan for c in FEATURE_COLUMNS}


# ---------------------------------------------------------------- camera
def camera_features(path: Path, ctx: dict) -> dict[str, float]:
    """Brightness, contrast and sharpness (variance of the Laplacian) of a frame.

    ctx: expected `width`, `height` and `image_feature_width`.
    """
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            if ctx.get("width") and (w, h) != (ctx["width"], ctx["height"]):
                raise CorruptFileError(f"size {w}x{h} != expected {ctx['width']}x{ctx['height']}")
            target_w = ctx.get("image_feature_width", 320)
            gray = im.convert("L").resize((target_w, max(1, round(h * target_w / w))))
    except (UnidentifiedImageError, OSError) as e:
        raise CorruptFileError(str(e)) from e

    g = np.asarray(gray, dtype=np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    out = _empty()
    out.update(
        brightness=float(g.mean()),
        contrast=float(g.std()),
        sharpness=float(lap.var()),
    )
    return out


# ---------------------------------------------------------------- lidar
LIDAR_DIMS = 5  # nuScenes .pcd.bin: x, y, z, intensity, ring index (float32)


def load_lidar(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) == 0 or len(raw) % (4 * LIDAR_DIMS) != 0:
        raise CorruptFileError(f"{len(raw)} bytes is not a multiple of {4 * LIDAR_DIMS}")
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, LIDAR_DIMS)


def lidar_features(path: Path, ctx: dict) -> dict[str, float]:
    pts = load_lidar(path)
    rng = np.linalg.norm(pts[:, :3], axis=1)
    out = _empty()
    out.update(
        n_points=float(len(pts)),
        mean_range=float(rng.mean()) if len(pts) else 0.0,
        max_range=float(rng.max()) if len(pts) else 0.0,
        mean_intensity=float(pts[:, 3].mean()) if len(pts) else 0.0,
    )
    return out


# ---------------------------------------------------------------- radar
def parse_pcd_header(raw: bytes) -> tuple[dict[str, list[str]], int]:
    """Parse a PCD header. Returns (fields, byte offset where data starts)."""
    header: dict[str, list[str]] = {}
    offset = 0
    for line in raw.split(b"\n"):
        offset += len(line) + 1
        parts = line.decode("ascii", errors="replace").strip().split()
        if not parts or parts[0].startswith("#"):
            continue
        header[parts[0]] = parts[1:]
        if parts[0] == "DATA":
            return header, offset
        if offset > 4096:
            break
    raise CorruptFileError("no DATA line in PCD header")


def radar_features(path: Path, ctx: dict) -> dict[str, float]:
    raw = path.read_bytes()
    header, offset = parse_pcd_header(raw)
    try:
        n = int(header["POINTS"][0])
        record = sum(int(s) * int(c) for s, c in zip(header["SIZE"], header["COUNT"], strict=True))
    except (KeyError, ValueError) as e:
        raise CorruptFileError(f"bad PCD header: {e}") from e
    if header.get("DATA", [""])[0] == "binary" and len(raw) - offset != n * record:
        raise CorruptFileError(f"expected {n * record} data bytes, found {len(raw) - offset}")
    out = _empty()
    out.update(n_points=float(n))
    return out


REGISTRY: dict[str, Modality] = {
    "camera": Modality("camera", ".jpg", camera_features),
    "lidar": Modality("lidar", ".pcd.bin", lidar_features),
    "radar": Modality("radar", ".pcd", radar_features),
}


def get_modality(name: str) -> Modality:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"No handler registered for modality '{name}'. "
            f"Add one to avdata.sensors.REGISTRY (known: {sorted(REGISTRY)})."
        ) from None
