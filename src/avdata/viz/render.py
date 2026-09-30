"""Rendering of camera + LiDAR scenes with 2D/3D boxes (Pillow only).

Used offline (report gallery, scene animations, failure gallery) and online
(demo API), so it deliberately depends on nothing heavier than Pillow and
NumPy.

Colour code, the same in every view:
  ground truth        aqua
  prediction (LiDAR)  blue
  prediction (mono)   orange (too few LiDAR points; depth from box height)
  LiDAR points        depth, near = yellow, far = purple
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avdata import geometry
from avdata.fusion.frames import FusionFrame, global_to_ego, project, yaw_quat

GT = (27, 175, 122)
PRED = (42, 120, 214)
MONO = (235, 104, 52)
BG = (17, 17, 16)
INK = (240, 240, 236)
MUTED = (150, 150, 145)
GRID = (60, 60, 58)
DEPTH_LUT = np.array(
    [(253, 231, 37), (94, 201, 98), (33, 145, 140), (59, 82, 139), (68, 1, 84)], dtype=np.float64
)  # near -> far
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


@lru_cache(maxsize=8)
def font(size: int = 14, bold: bool = False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for n in names:
        for base in ("/usr/share/fonts/truetype/dejavu/", ""):
            try:
                return ImageFont.truetype(base + n, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def depth_colors(depth: np.ndarray, dmax: float = 60.0) -> np.ndarray:
    t = np.clip(depth / dmax, 0, 1) * (len(DEPTH_LUT) - 1)
    i = np.minimum(t.astype(int), len(DEPTH_LUT) - 2)
    f = (t - i)[:, None]
    return (DEPTH_LUT[i] * (1 - f) + DEPTH_LUT[i + 1] * f).astype(np.uint8)


def _box_fields(b) -> tuple[list, list, float, str, float, str]:
    """(center, size, yaw, cls, score, source) from a Box3D or a dict."""
    if isinstance(b, dict):
        return (
            b["center"],
            b["size"],
            b["yaw"],
            b["cls"],
            b.get("score", 1.0),
            b.get("source", "gt"),
        )
    return b.center, b.size, b.yaw, b.cls, b.score, b.source


def _corners_global(center, size, yaw) -> np.ndarray:
    return geometry.box_corners(np.asarray(center), np.asarray(size), np.asarray(yaw_quat(yaw)))


# ------------------------------------------------------------------ camera view
def render_camera(
    image: Image.Image,
    fr: FusionFrame,
    gt=(),
    preds=(),
    lidar: bool = True,
    labels: bool = True,
    dmax: float = 60.0,
) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    if lidar:
        uv, depth, inside = project(fr.points_cam, fr.K, fr.width, fr.height)
        cols = depth_colors(depth[inside], dmax)
        for (u, v), c in zip(uv[:, inside].T, cols, strict=False):
            draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(*map(int, c), 230))
    for boxes, is_gt in ((gt, True), (preds, False)):
        for b in boxes:
            center, size, yaw, cls, score, source = _box_fields(b)
            color = GT if is_gt else (MONO if source == "mono" else PRED)
            cam = geometry.global_to_sensor(
                _corners_global(center, size, yaw),
                fr.cam.ego_t,
                fr.cam.ego_q,
                fr.cam.sensor_t,
                fr.cam.sensor_q,
            )
            if (cam[2] < 0.5).all():
                continue
            uv = geometry.project_to_image(np.where(cam[2] > 0.5, cam, np.nan), fr.K)
            w = 3 if is_gt else 2
            for a, c in EDGES:
                if cam[2, a] > 0.5 and cam[2, c] > 0.5:
                    draw.line([tuple(uv[:, a]), tuple(uv[:, c])], fill=color, width=w)
            if cam[2, :4].min() > 0.5:  # mark the front face
                draw.line([tuple(uv[:, 0]), tuple(uv[:, 2])], fill=(*color, 150), width=1)
                draw.line([tuple(uv[:, 1]), tuple(uv[:, 3])], fill=(*color, 150), width=1)
            if labels and not is_gt:
                x, y = np.nanmin(uv[0]), np.nanmin(uv[1])
                if np.isfinite(x) and np.isfinite(y):
                    txt = f"{cls} {score:.2f}"
                    f = font(15)
                    tw = draw.textlength(txt, font=f)
                    draw.rectangle([x, y - 20, x + tw + 8, y], fill=(*color, 220))
                    draw.text((x + 4, y - 19), txt, font=f, fill=(255, 255, 255))
    return img


# ------------------------------------------------------------------ bird's-eye view
def render_bev(
    fr: FusionFrame, gt=(), preds=(), range_m: float = 60.0, size: int = 600
) -> Image.Image:
    """Top-down view in the ego frame at the camera timestamp (x forward = up)."""
    span = range_m + 10
    s = size / span

    def px(x, y):
        return (span / 2 - y) * s, (range_m - x) * s

    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img, "RGBA")
    for r in range(10, int(range_m) + 1, 10):
        x0, y0 = px(r, r)
        x1, y1 = px(-r, -r)
        d.ellipse([x0, y0, x1, y1], outline=GRID if r % 20 else (85, 85, 82), width=1)
        if r % 20 == 0:
            tx, ty = px(r, 0)
            d.text((tx + 3, ty + 1), f"{r} m", font=font(11), fill=MUTED)
    # camera field of view
    cam_xy = np.asarray(fr.cam.sensor_t[:2])
    axis = geometry.quat_to_rot(fr.cam.sensor_q)[:, 2]
    heading = np.arctan2(axis[1], axis[0])
    half = np.arctan(fr.width / 2 / fr.K[0, 0])
    for a in (heading - half, heading + half):
        end = cam_xy + np.array([np.cos(a), np.sin(a)]) * range_m * 1.5
        d.line([px(*cam_xy), px(*end)], fill=(120, 120, 115, 160), width=1)
    # LiDAR points coloured by height
    pe = fr.points_ego
    keep = (pe[0] > -10) & (pe[0] < range_m) & (np.abs(pe[1]) < span / 2)
    z = pe[2, keep]
    cols = depth_colors(np.clip(3.0 - z, 0, 4), 4.0)
    for (x, y), c in zip(pe[:2, keep].T, cols, strict=False):
        u, v = px(x, y)
        d.point((u, v), fill=tuple(int(k) for k in c))
    # ego vehicle
    ego = [px(2.5, 1.0), px(2.5, -1.0), px(-1.5, -1.0), px(-1.5, 1.0)]
    d.polygon(ego, fill=(230, 230, 225))
    for boxes, is_gt in ((gt, True), (preds, False)):
        for b in boxes:
            center, sz, yaw, _cls, _score, source = _box_fields(b)
            corners = global_to_ego(_corners_global(center, sz, yaw), fr.cam)
            poly = [px(*corners[:2, i]) for i in (0, 1, 5, 4)]
            color = GT if is_gt else (MONO if source == "mono" else PRED)
            d.polygon(poly, outline=color, width=3 if is_gt else 2)
            c = corners[:2].mean(1)
            front = corners[:2, :4].mean(1)
            d.line([px(*c), px(*front)], fill=color, width=2)
    y = 8
    for label, color in (
        ("ground truth", GT),
        ("fused 3D (LiDAR)", PRED),
        ("fused 3D (mono)", MONO),
    ):
        d.rectangle([10, y + 3, 22, y + 13], outline=color, width=2)
        d.text((28, y), label, font=font(12), fill=INK)
        y += 18
    return img


# ------------------------------------------------------------------ composite
def compose(
    camera: Image.Image, bev: Image.Image, title: str, subtitle: str = "", height: int = 600
) -> Image.Image:
    cam = camera.resize((round(camera.width * height / camera.height), height), Image.BILINEAR)
    bev = bev.resize((height, height), Image.BILINEAR)
    head = 52
    out = Image.new("RGB", (cam.width + bev.width, height + head), BG)
    out.paste(cam, (0, head))
    out.paste(bev, (cam.width, head))
    d = ImageDraw.Draw(out)
    d.text((14, 7), title, font=font(18, bold=True), fill=INK)
    d.text((14, 30), subtitle, font=font(13), fill=MUTED)
    return out


def save_gif(frames: list[Image.Image], path: Path, ms: int = 500, width: int = 1100) -> Path:
    if not frames:
        return path
    small = [f.resize((width, round(f.height * width / f.width)), Image.BILINEAR) for f in frames]
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=192) for f in small]
    path.parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=ms, loop=0, optimize=True)
    return path
