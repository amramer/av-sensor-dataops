"""Camera + LiDAR late fusion: lift 2D detections to 3D boxes (frustum method).

For every 2D box from the camera detector:

1. take the LiDAR points whose projection falls inside the (slightly shrunk)
   box: the viewing frustum of the object
2. drop ground points (height above the estimated ground plane)
3. split the remaining points by depth into clusters and keep the one that
   best combines size (number of points) with agreement to the depth expected
   from the box height and the class height; this separates the object from
   background seen past its edges and from occluding objects in front of it
4. place the box: the nearest cluster points mark the visible surface, so the
   centre lies half the object's extent further along the viewing ray
5. orientation from the principal axis of the points seen from above (with a
   check whether that axis is the object's length or width), otherwise the
   ego heading
6. size from the class prior; height from the ground plane

When a box contains too few LiDAR points (far or occluded objects) the depth
is estimated from the box height and the prior object height (monocular
fallback); such boxes are marked `source="mono"` and get a lower score.

This is a transparent, training-free baseline. It shows the full multi-sensor
path (calibration, time alignment, projection, 3D evaluation) that a learned
fusion detector (e.g. BEVFusion, CenterPoint + camera) plugs into.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from avdata import geometry
from avdata.fusion.frames import FusionFrame, ego_to_global, project, quat_yaw

ELONGATED_MIN_POINTS = 10


@dataclass
class LiftParams:
    box_shrink: float = 0.1
    ground_offset_m: float = 0.3
    min_points: int = 3
    cluster_gap_m: float = 0.8
    near_percentile: float = 10.0
    depth_prior_sigma: float = 0.2  # relative uncertainty of the depth from box height
    mono_score_factor: float = 0.5


@dataclass
class Det2D:
    cls: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Box3D:
    cls: str
    score: float
    center: list[float]  # x, y, z
    size: list[float]  # w, l, h
    yaw: float
    n_points: int
    source: str  # "lidar" | "mono"
    frame: str = "ego"
    box2d: list[float] = field(default_factory=list)

    def to_global(self, fr: FusionFrame) -> Box3D:
        if self.frame == "global":
            return self
        c = ego_to_global(np.reshape(self.center, (3, 1)), fr.cam)[:, 0]
        yaw = self.yaw + quat_yaw(fr.cam.ego_q)
        return Box3D(
            self.cls,
            self.score,
            c.tolist(),
            list(self.size),
            wrap(yaw),
            self.n_points,
            self.source,
            "global",
            list(self.box2d),
        )

    def as_dict(self) -> dict:
        d = asdict(self)
        d["center"] = [round(v, 3) for v in d["center"]]
        d["size"] = [round(v, 3) for v in d["size"]]
        d["yaw"] = round(d["yaw"], 4)
        d["score"] = round(d["score"], 4)
        d["box2d"] = [round(v, 1) for v in d["box2d"]]
        return d


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def estimate_ground(points_ego: np.ndarray, radius: float = 40.0) -> float:
    near = np.hypot(points_ego[0], points_ego[1]) < radius
    z = points_ego[2, near] if near.any() else points_ego[2]
    return float(np.percentile(z, 5)) if z.size else 0.0


def dominant_cluster(
    depths: np.ndarray,
    gap: float,
    expected: float | None = None,
    rel_sigma: float = 0.2,
    near_pct: float = 10.0,
) -> np.ndarray:
    """Indices (into depths) of the best depth cluster.

    Clusters are runs of sorted depths without a gap larger than `gap`. Each is
    scored by its size, weighted by how well its near-surface depth agrees with
    the `expected` depth (from the 2D box height, which is set by the object's
    nearest part). Without an expectation the
    largest cluster wins; ties go to the nearest.
    """
    order = np.argsort(depths)
    d = depths[order]
    cuts = np.where(np.diff(d) > gap)[0] + 1
    groups = np.split(order, cuts)

    def score(g):
        s = float(len(g))
        if expected:
            dev = (np.percentile(depths[g], near_pct) - expected) / (rel_sigma * expected)
            s *= float(np.exp(-0.5 * dev * dev))
        return (s, -depths[g].min())

    return max(groups, key=score)


def lift(
    fr: FusionFrame,
    dets: list[Det2D],
    sizes: dict,
    lp: LiftParams | None = None,
    timings: dict | None = None,
) -> list[Box3D]:
    lp = lp or LiftParams()
    t0 = time.perf_counter()
    pts_cam = fr.points_cam
    pts_ego = fr.points_ego
    uv, depth, inside = project(pts_cam, fr.K, fr.width, fr.height)
    ground = estimate_ground(pts_ego)
    above = pts_ego[2] > ground + lp.ground_offset_m
    cand = inside & above
    cam_xy = np.asarray(fr.cam.sensor_t[:2])
    t1 = time.perf_counter()

    boxes = []
    for d in dets:
        w, length, h = sizes.get(d.cls, (1.0, 1.0, 1.0))
        x1, y1, x2, y2 = d.box
        sx, sy = (x2 - x1) * lp.box_shrink / 2, (y2 - y1) * lp.box_shrink / 2
        m = cand & (uv[0] >= x1 + sx) & (uv[0] <= x2 - sx) & (uv[1] >= y1 + sy) & (uv[1] <= y2 - sy)
        idx = np.where(m)[0]
        z_expected = fr.K[1, 1] * h / max(y2 - y1, 1.0)  # depth from box height (pinhole)
        sel = (
            idx[
                dominant_cluster(
                    depth[idx],
                    lp.cluster_gap_m,
                    z_expected,
                    lp.depth_prior_sigma,
                    lp.near_percentile,
                )
            ]
            if len(idx)
            else idx
        )
        if len(sel) >= lp.min_points:
            bev = pts_ego[:2, sel]
            yaw = _yaw_from_points(bev, w, length)
            ray = bev.mean(1) - cam_xy
            ray /= np.linalg.norm(ray) + 1e-9
            along = (bev - cam_xy[:, None]).T @ ray
            normal = np.array([-ray[1], ray[0]])
            lateral = float(np.median((bev - cam_xy[:, None]).T @ normal))
            theta = np.arctan2(ray[1], ray[0]) - yaw
            extent = abs(np.cos(theta)) * length / 2 + abs(np.sin(theta)) * w / 2
            dist = float(np.percentile(along, lp.near_percentile)) + extent
            xy = cam_xy + ray * dist + normal * lateral
            score, n, source = d.score, len(sel), "lidar"
        else:
            # monocular fallback: similar triangles with the prior height
            z = z_expected
            u = (x1 + x2) / 2
            x_cam = (u - fr.K[0, 2]) * z / fr.K[0, 0]
            pc = np.array([[x_cam], [0.0], [z]])
            R = geometry.quat_to_rot(fr.cam.sensor_q)
            pe = R @ pc + np.reshape(fr.cam.sensor_t, (3, 1))
            ray = pe[:2, 0] - cam_xy
            ray /= np.linalg.norm(ray) + 1e-9
            extent = (length + w) / 4
            xy = pe[:2, 0] + ray * extent
            yaw, score, n, source = 0.0, d.score * lp.mono_score_factor, len(idx), "mono"
        boxes.append(
            Box3D(
                d.cls,
                float(score),
                [float(xy[0]), float(xy[1]), ground + h / 2],
                [w, length, h],
                wrap(yaw),
                int(n),
                source,
                "ego",
                list(d.box),
            )
        )
    if timings is not None:
        timings["project_ms"] = (t1 - t0) * 1e3
        timings["lift_ms"] = (time.perf_counter() - t1) * 1e3
    return boxes


def _yaw_from_points(bev: np.ndarray, w: float, length: float) -> float:
    """Heading from the principal axis of the object's points seen from above.

    One visible face gives an axis along that face, which may be the width
    (rear of a car) or the length (side of a car). The measured extent along
    the axis decides which one it is. Falls back to the ego heading (0).
    """
    if bev.shape[1] < ELONGATED_MIN_POINTS or max(w, length) / max(min(w, length), 1e-6) < 1.3:
        return 0.0
    c = bev - bev.mean(1, keepdims=True)
    evals, evecs = np.linalg.eigh(np.cov(c))
    if evals[1] < 2.0 * max(evals[0], 1e-6):
        return 0.0
    axis = evecs[:, 1]
    proj = axis @ c
    ext = float(np.percentile(proj, 95) - np.percentile(proj, 5))
    angle = float(np.arctan2(axis[1], axis[0]))
    long_dim_is_length = length >= w
    axis_is_long_dim = abs(ext - max(w, length)) <= abs(ext - min(w, length))
    if axis_is_long_dim != long_dim_is_length:
        angle += np.pi / 2
    # heading is ambiguous by pi; choose the one closest to the ego heading
    angle = wrap(angle)
    if abs(angle) > np.pi / 2:
        angle = wrap(angle + np.pi)
    return angle
