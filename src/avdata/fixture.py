"""Synthetic dataset in the exact nuScenes on-disk format.

nuScenes needs a registration to download, so tests and CI cannot fetch it.
This module writes a small dataset with the same folder layout, the same
13 JSON tables, the same file formats (.jpg, .pcd.bin, .pcd) and realistic
sensor rates, so the whole pipeline can run end to end in seconds.

LiDAR sweeps are simulated, not random: a 32-ring scanner hits the ground,
background structures and the visible faces of every annotated object, so
`num_lidar_pts` is real and camera+LiDAR fusion can be tested end to end.

`faults=True` injects the problems the quality stage must catch:
a truncated lidar file, a missing camera file, a camera/lidar sync error,
dropped lidar sweeps, weak labels and a wrong camera calibration.
"""

from __future__ import annotations

import json
import random
import struct
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from avdata import geometry

CAM_INTRINSIC = [[1266.4, 0.0, 816.3], [0.0, 1266.4, 491.5], [0.0, 0.0, 1.0]]
IMG_W, IMG_H = 1600, 900
SENSORS = {
    # channel: (modality, translation, rotation[w,x,y,z], rate_hz)
    "CAM_FRONT": ("camera", [1.70, 0.0, 1.51], [0.5, -0.5, 0.5, -0.5], 12.0),
    "LIDAR_TOP": ("lidar", [0.94, 0.0, 1.84], [0.7071068, 0.0, 0.0, -0.7071068], 20.0),
    "RADAR_FRONT": ("radar", [3.41, 0.0, 0.52], [1.0, 0.0, 0.0, 0.0], 13.0),
}
CATEGORIES = [
    ("vehicle.car", (40, 90, 200), (1.9, 4.5, 1.6)),
    ("human.pedestrian.adult", (220, 60, 60), (0.7, 0.7, 1.75)),
    ("vehicle.bicycle", (60, 180, 80), (0.6, 1.8, 1.3)),
    ("movable_object.barrier", (230, 200, 40), (2.0, 0.5, 1.0)),
    ("animal", (120, 80, 40), (0.4, 0.8, 0.5)),
]
SCENES = [
    # name, description, location, speed m/s, heading rad
    (
        "scene-9001",
        "Day, parked cars, peds crossing, bicycle on bike lane",
        "boston-seaport",
        6.0,
        0.0,
    ),
    (
        "scene-9002",
        "Night, big street, parked cars, ped on sidewalk",
        "singapore-onenorth",
        4.0,
        0.6,
    ),
    ("scene-9003", "Rain, waiting at intersection, barrier", "singapore-queenstown", 0.0, -0.4),
]
LIDAR_RINGS = np.linspace(-30.0, 10.0, 32)  # elevation of each laser ring, degrees
LIDAR_DENSITY = 2500.0  # object points per m^2 of visible surface at 1 m range
RADAR_FIELDS = (
    "x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state "
    "x_rms y_rms invalid_state pdh0 vx_rms vy_rms"
)
RADAR_SIZE = [4, 4, 4, 1, 2, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1]
RADAR_TYPE = "F F F I I F F F F F I I I I I I I I"


def _tok() -> str:
    return uuid.uuid4().hex


@dataclass
class _Tables:
    t: dict[str, list[dict]] = field(
        default_factory=lambda: {
            k: []
            for k in [
                "attribute",
                "calibrated_sensor",
                "category",
                "ego_pose",
                "instance",
                "log",
                "map",
                "sample",
                "sample_annotation",
                "sample_data",
                "scene",
                "sensor",
                "visibility",
            ]
        }
    )

    def add(self, table: str, row: dict) -> dict:
        self.t[table].append(row)
        return row


def _link(rows: list[dict]) -> None:
    for i, r in enumerate(rows):
        r["prev"] = rows[i - 1]["token"] if i > 0 else ""
        r["next"] = rows[i + 1]["token"] if i < len(rows) - 1 else ""


def _yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def _visible_face_points(center, size, rotation, sensor_xyz, rng) -> np.ndarray:
    """Points sampled on the faces of a 3D box that face the sensor (global frame, 3xN)."""
    w, length, h = size
    R = geometry.quat_to_rot(rotation)
    c = np.asarray(center, dtype=np.float64)
    out = []
    # face: (normal axis, sign, extent along the two in-plane axes)
    half = np.array([length / 2, w / 2, h / 2])
    for axis in range(3):
        for sign in (-1.0, 1.0):
            normal = R[:, axis] * sign
            face_c = c + normal * half[axis]
            if axis == 2 and sign < 0:
                continue  # bottom face touches the ground
            to_sensor = np.asarray(sensor_xyz) - face_c
            if normal @ to_sensor <= 0:
                continue
            u, v = [a for a in range(3) if a != axis]
            area = 4 * half[u] * half[v]
            dist = max(np.linalg.norm(to_sensor), 1.0)
            n = int(min(400, LIDAR_DENSITY * area / dist**2))
            if n == 0:
                continue
            su = rng.uniform(-half[u], half[u], n)
            sv = rng.uniform(-half[v], half[v], n)
            pts = face_c[:, None] + R[:, u][:, None] * su + R[:, v][:, None] * sv
            out.append(pts)
    return np.hstack(out) if out else np.zeros((3, 0))


def _simulate_lidar(objects, ego_t, ego_q, n_total, rng) -> tuple[np.ndarray, dict[str, int]]:
    """One LiDAR sweep in the sensor frame (N x 5: x, y, z, intensity, ring)."""
    _, l_t, l_q, _ = SENSORS["LIDAR_TOP"]
    R_e = geometry.quat_to_rot(ego_q)
    sensor_g = R_e @ np.asarray(l_t) + np.asarray(ego_t)
    h = sensor_g[2]

    obj_pts, counts = [], {}
    for o in objects:
        p = _visible_face_points(o["center"], o["size"], o["rotation"], sensor_g, rng)
        counts[o["instance"]["token"]] = p.shape[1]
        obj_pts.append(p)
    obj = np.hstack(obj_pts) if obj_pts else np.zeros((3, 0))

    # background structures (buildings, poles) 35-70 m away, all around the car
    n_wall = 800
    a = rng.uniform(-np.pi, np.pi, n_wall)
    r = rng.uniform(35, 70, n_wall)
    walls = np.vstack(
        [sensor_g[0] + r * np.cos(a), sensor_g[1] + r * np.sin(a), rng.uniform(0.3, 6.0, n_wall)]
    )
    # ground returns: downward rings hit the road at range h / tan(-elevation)
    n_ground = max(n_total - obj.shape[1] - n_wall, 0)
    down = LIDAR_RINGS[LIDAR_RINGS < -1.5]
    elev = np.deg2rad(rng.choice(down, n_ground))
    rg = h / np.tan(-elev)
    a = rng.uniform(-np.pi, np.pi, n_ground)
    ground = np.vstack(
        [sensor_g[0] + rg * np.cos(a), sensor_g[1] + rg * np.sin(a), np.zeros(n_ground)]
    )
    pts_g = np.hstack([obj, walls, ground])
    pts_g[:2] += rng.normal(0, 0.02, pts_g[:2].shape)  # range noise

    pts_l = geometry.global_to_sensor(pts_g, ego_t, ego_q, l_t, l_q)
    elev_deg = np.rad2deg(np.arctan2(pts_l[2], np.hypot(pts_l[0], pts_l[1])))
    ring = np.abs(elev_deg[:, None] - LIDAR_RINGS[None]).argmin(1)
    out = np.zeros((pts_l.shape[1], 5), dtype=np.float32)
    out[:, :3] = pts_l.T
    out[:, 3] = rng.uniform(0, 100, pts_l.shape[1])
    out[:, 4] = ring
    return out, counts


def _write_radar(path: Path, n: int) -> None:
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        f"FIELDS {RADAR_FIELDS}\nSIZE {' '.join(map(str, RADAR_SIZE))}\nTYPE {RADAR_TYPE}\n"
        f"COUNT {' '.join(['1'] * len(RADAR_SIZE))}\nWIDTH {n}\nHEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    )
    fmt = "<fffbhfffffbbbbbbbb"
    body = b"".join(
        struct.pack(fmt, 10.0 + i, 0.5, 0.0, 1, i, 5.0, 0, 0, 0, 0, 1, 3, 1, 1, 0, 1, 0, 0)
        for i in range(n)
    )
    path.write_bytes(header.encode() + body)


def make_fixture(
    root: Path,
    samples_per_scene: int = 5,
    lidar_points: int = 6000,
    seed: int = 0,
    faults: bool = False,
    version: str = "v1.0-mini",
) -> Path:
    """Write a nuScenes-format dataset under `root`. Returns `root`."""
    random.seed(seed)
    rng = np.random.default_rng(seed)
    root = Path(root)
    tb = _Tables()

    for i in range(1, 5):
        tb.add(
            "visibility",
            {
                "token": str(i),
                "level": ["v0-40", "v40-60", "v60-80", "v80-100"][i - 1],
                "description": f"visibility bin {i}",
            },
        )
    attr_moving = tb.add(
        "attribute", {"token": _tok(), "name": "vehicle.parked", "description": ""}
    )
    cats = {
        name: tb.add("category", {"token": _tok(), "name": name, "description": ""})
        for name, _, _ in CATEGORIES
    }
    sensor_rows = {
        ch: tb.add("sensor", {"token": _tok(), "channel": ch, "modality": mod})
        for ch, (mod, *_rest) in SENSORS.items()
    }

    t_start = 1_531_883_530_000_000  # microseconds, like nuScenes
    for s_idx, (scene_name, desc, location, speed, heading) in enumerate(SCENES):
        log = tb.add(
            "log",
            {
                "token": _tok(),
                "logfile": f"n015-2018-07-18-{scene_name}",
                "vehicle": "n015",
                "date_captured": "2018-07-18",
                "location": location,
            },
        )
        tb.add(
            "map",
            {
                "token": _tok(),
                "log_tokens": [log["token"]],
                "category": "semantic_prior",
                "filename": f"maps/{location}.png",
            },
        )
        calib = {}
        for ch, (_mod, trans, rot, _hz) in SENSORS.items():
            calib[ch] = tb.add(
                "calibrated_sensor",
                {
                    "token": _tok(),
                    "sensor_token": sensor_rows[ch]["token"],
                    "translation": trans,
                    "rotation": rot,
                    "camera_intrinsic": CAM_INTRINSIC if ch.startswith("CAM") else [],
                },
            )

        t0 = t_start + s_idx * 60_000_000
        ego_x0, ego_y0 = 100.0 * (s_idx + 1), 50.0
        night = "night" in desc.lower()

        def ego_at(
            t_us: int,
            _x0: float = ego_x0,
            _y0: float = ego_y0,
            _v: float = speed,
            _t0: int = t0,
            _h: float = heading,
        ) -> tuple[list, list]:
            dt = (t_us - _t0) / 1e6
            pos = [_x0 + _v * dt * np.cos(_h), _y0 + _v * dt * np.sin(_h), 0.0]
            return [float(v) for v in pos], _yaw_quat(_h)

        # --- objects: static boxes ahead of the ego vehicle
        objects = []
        cat_pool = [
            "vehicle.car",
            "vehicle.car",
            "human.pedestrian.adult",
            "movable_object.barrier",
        ]
        if "bicycle" in desc:
            cat_pool.append("vehicle.bicycle")
        if s_idx == 1:
            cat_pool += ["human.pedestrian.adult"] * 4 + ["animal"]
        for k, cname in enumerate(cat_pool):
            size = next(sz for n, _, sz in CATEGORIES if n == cname)
            fwd, lat = 18 + 7 * k, rng.uniform(-4, 4)
            center = [
                ego_x0 + fwd * np.cos(heading) - lat * np.sin(heading),
                ego_y0 + fwd * np.sin(heading) + lat * np.cos(heading),
                size[2] / 2,
            ]
            center = [float(v) for v in center]
            yaw = heading + rng.uniform(-0.3, 0.3)
            objects.append(
                {
                    "instance": tb.add(
                        "instance",
                        {
                            "token": _tok(),
                            "category_token": cats[cname]["token"],
                            "nbr_annotations": 0,
                            "first_annotation_token": "",
                            "last_annotation_token": "",
                        },
                    ),
                    "category": cname,
                    "center": center,
                    "size": list(size),
                    "rotation": _yaw_quat(yaw),
                }
            )

        # --- samples (keyframes at 2 Hz)
        samples = []
        for j in range(samples_per_scene):
            samples.append(
                tb.add(
                    "sample", {"token": _tok(), "timestamp": t0 + j * 500_000, "scene_token": ""}
                )
            )
        _link(samples)
        scene = tb.add(
            "scene",
            {
                "token": _tok(),
                "log_token": log["token"],
                "nbr_samples": len(samples),
                "first_sample_token": samples[0]["token"],
                "last_sample_token": samples[-1]["token"],
                "name": scene_name,
                "description": desc,
            },
        )
        for smp in samples:
            smp["scene_token"] = scene["token"]

        # --- annotations
        ann_by_instance: dict[str, list[dict]] = {o["instance"]["token"]: [] for o in objects}
        ann_index: dict[tuple[str, str], dict] = {}
        for smp in samples:
            for o in objects:
                ann = tb.add(
                    "sample_annotation",
                    {
                        "token": _tok(),
                        "sample_token": smp["token"],
                        "instance_token": o["instance"]["token"],
                        "visibility_token": "4"
                        if o["category"] != "movable_object.barrier"
                        else "3",
                        "attribute_tokens": [attr_moving["token"]]
                        if o["category"] == "vehicle.car"
                        else [],
                        "translation": o["center"],
                        "size": o["size"],
                        "rotation": o["rotation"],
                        "num_lidar_pts": 0,  # set from the simulated LiDAR keyframe below
                        "num_radar_pts": int(rng.integers(0, 5)),
                    },
                )
                ann_by_instance[o["instance"]["token"]].append(ann)
                ann_index[(smp["token"], o["instance"]["token"])] = ann
        for o in objects:
            anns = ann_by_instance[o["instance"]["token"]]
            _link(anns)
            o["instance"].update(
                nbr_annotations=len(anns),
                first_annotation_token=anns[0]["token"],
                last_annotation_token=anns[-1]["token"],
            )

        # --- sensor frames (keyframes + sweeps at each sensor's rate)
        t_end = samples[-1]["timestamp"]
        for ch, (mod, _trans, _rot, hz) in SENSORS.items():
            period = int(1e6 / hz)
            times = list(range(t0, t_end + 1, period))
            key_times = {smp["timestamp"]: smp for smp in samples}
            frames = []
            for t in times:
                nearest = min(samples, key=lambda s, t=t: abs(s["timestamp"] - t))
                is_key = (
                    abs(nearest["timestamp"] - t) < period / 2 and nearest["timestamp"] in key_times
                )
                # a sensor keyframe is the frame closest to the sample time
                if is_key and any(
                    f["sample_token"] == nearest["token"] and f["is_key_frame"] for f in frames
                ):
                    is_key = False
                folder = "samples" if is_key else "sweeps"
                ext = {"camera": ".jpg", "lidar": ".pcd.bin", "radar": ".pcd"}[mod]
                fname = f"{folder}/{ch}/n015-2018-07-18__{ch}__{t}{ext}"
                trans, rot = ego_at(t)
                ego = tb.add(
                    "ego_pose",
                    {"token": _tok(), "timestamp": t, "translation": trans, "rotation": rot},
                )
                frames.append(
                    tb.add(
                        "sample_data",
                        {
                            "token": _tok(),
                            "sample_token": nearest["token"],
                            "ego_pose_token": ego["token"],
                            "calibrated_sensor_token": calib[ch]["token"],
                            "timestamp": t,
                            "fileformat": ext.strip(".").replace("pcd.bin", "pcd"),
                            "is_key_frame": is_key,
                            "height": IMG_H if mod == "camera" else 0,
                            "width": IMG_W if mod == "camera" else 0,
                            "filename": fname,
                        },
                    )
                )
            _link(frames)

            for f in frames:
                path = root / f["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                if mod == "lidar":
                    trans, rot = ego_at(f["timestamp"])
                    pts, counts = _simulate_lidar(objects, trans, rot, lidar_points, rng)
                    path.write_bytes(pts.tobytes())
                    if f["is_key_frame"]:
                        for inst, n in counts.items():
                            ann_index[(f["sample_token"], inst)]["num_lidar_pts"] = int(n)
                elif mod == "radar":
                    _write_radar(path, 12)
                else:
                    bg = (30, 32, 40) if night else (125, 130, 135)
                    img = Image.new("RGB", (IMG_W, IMG_H), bg)
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([0, IMG_H * 0.55, IMG_W, IMG_H], fill=tuple(c // 2 for c in bg))
                    trans, rot = ego_at(f["timestamp"])
                    # painter's order: far objects first so near ones occlude them
                    for o in sorted(
                        objects,
                        key=lambda o: (
                            -np.hypot(o["center"][0] - trans[0], o["center"][1] - trans[1])
                        ),
                    ):
                        corners = geometry.box_corners(
                            np.array(o["center"]), np.array(o["size"]), np.array(o["rotation"])
                        )
                        cam = geometry.global_to_sensor(
                            corners, trans, rot, SENSORS[ch][1], SENSORS[ch][2]
                        )
                        box = geometry.box_to_2d(cam, np.array(CAM_INTRINSIC), IMG_W, IMG_H)
                        if box:
                            color = next(c for n, c, _ in CATEGORIES if n == o["category"])
                            draw.rectangle(box, fill=color, outline=(255, 255, 255), width=3)
                    img.save(path, quality=85)

    if faults:
        _inject_faults(root, tb)

    tables_dir = root / version
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tb.t.items():
        (tables_dir / f"{name}.json").write_text(json.dumps(rows, indent=0))
    return root


def _inject_faults(root: Path, tb: _Tables) -> None:
    sd = tb.t["sample_data"]
    calib_ch = {
        c["token"]: next(s["channel"] for s in tb.t["sensor"] if s["token"] == c["sensor_token"])
        for c in tb.t["calibrated_sensor"]
    }
    lidar = [f for f in sd if calib_ch[f["calibrated_sensor_token"]] == "LIDAR_TOP"]
    cams = [f for f in sd if calib_ch[f["calibrated_sensor_token"]] == "CAM_FRONT"]

    # 1. corrupt lidar file (truncated mid-record)
    p = root / lidar[3]["filename"]
    p.write_bytes(p.read_bytes()[:1001])
    # 2. missing camera sweep file
    (root / next(f for f in cams if not f["is_key_frame"])["filename"]).unlink()
    # 3. camera keyframe out of sync by 120 ms
    key_cam = [f for f in cams if f["is_key_frame"]][2]
    key_cam["timestamp"] += 120_000
    # 4. dropped lidar sweeps: remove 4 consecutive sweep records (gap of ~250 ms)
    sweeps = [f for f in lidar if not f["is_key_frame"]]
    for f in sweeps[10:14]:
        sd.remove(f)
    # 5. weak labels: annotations without any lidar points
    for ann in tb.t["sample_annotation"][:3]:
        ann["num_lidar_pts"] = 0
    # 6. wrong camera calibration: the last scene's CAM_FRONT intrinsics have a focal
    #    length 40 % too short (e.g. the calibration file of another camera was used)
    cam_calibs = [c for c in tb.t["calibrated_sensor"] if calib_ch[c["token"]] == "CAM_FRONT"]
    bad = cam_calibs[-1]
    bad["camera_intrinsic"] = [
        [bad["camera_intrinsic"][0][0] * 0.6, 0.0, bad["camera_intrinsic"][0][2]],
        [0.0, bad["camera_intrinsic"][1][1] * 0.6, bad["camera_intrinsic"][1][2]],
        [0.0, 0.0, 1.0],
    ]
