"""Frames, poses and point transforms for camera + LiDAR fusion.

Everything the fusion needs for one camera image is bundled in a
`FusionFrame`: the image size and intrinsics, the camera and LiDAR poses at
their own capture times, and the raw LiDAR points in the LiDAR frame.
The same structure is built from the pipeline's silver tables and from an
API request, so offline evaluation and online serving run identical code.

Frames (nuScenes conventions)
  lidar   raw points as recorded
  ego     vehicle frame at a given timestamp (x forward, y left, z up)
  global  world frame; annotations live here
  camera  x right, y down, z forward (optical axis)
Because camera and LiDAR fire at different times while the car moves, points
go lidar -> ego(t_lidar) -> global -> ego(t_camera) -> camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avdata import geometry


@dataclass(frozen=True)
class SensorPose:
    """Sensor-to-ego calibration and ego-to-global pose at the sensor's capture time."""

    sensor_t: tuple[float, float, float]
    sensor_q: tuple[float, float, float, float]
    ego_t: tuple[float, float, float]
    ego_q: tuple[float, float, float, float]

    @classmethod
    def from_row(cls, r) -> SensorPose:
        return cls(
            tuple(map(float, r.sensor_translation)),
            tuple(map(float, r.sensor_rotation)),
            tuple(map(float, r.ego_translation)),
            tuple(map(float, r.ego_rotation)),
        )

    def as_dict(self) -> dict:
        return {k: list(getattr(self, k)) for k in ("sensor_t", "sensor_q", "ego_t", "ego_q")}

    @classmethod
    def from_dict(cls, d: dict) -> SensorPose:
        return cls(*(tuple(map(float, d[k])) for k in ("sensor_t", "sensor_q", "ego_t", "ego_q")))


@dataclass
class FusionFrame:
    K: np.ndarray  # 3x3 camera intrinsics
    width: int
    height: int
    cam: SensorPose
    lidar: SensorPose
    points_lidar: np.ndarray  # N x >=3, LiDAR frame

    # ---- cached views of the point cloud
    def __post_init__(self) -> None:
        self._global = None

    @property
    def points_global(self) -> np.ndarray:
        if self._global is None:
            p = self.points_lidar[:, :3].T.astype(np.float64)
            p = geometry.quat_to_rot(self.lidar.sensor_q) @ p + np.reshape(
                self.lidar.sensor_t, (3, 1)
            )
            p = geometry.quat_to_rot(self.lidar.ego_q) @ p + np.reshape(self.lidar.ego_t, (3, 1))
            self._global = p
        return self._global

    @property
    def points_ego(self) -> np.ndarray:
        """Points in the ego frame at the camera timestamp (3xN)."""
        return global_to_ego(self.points_global, self.cam)

    @property
    def points_cam(self) -> np.ndarray:
        return geometry.global_to_sensor(
            self.points_global, self.cam.ego_t, self.cam.ego_q, self.cam.sensor_t, self.cam.sensor_q
        )

    def calib_dict(self) -> dict:
        return {
            "camera_intrinsic": self.K.tolist(),
            "width": self.width,
            "height": self.height,
            "camera": self.cam.as_dict(),
            "lidar": self.lidar.as_dict(),
        }


def global_to_ego(p: np.ndarray, pose: SensorPose) -> np.ndarray:
    return geometry.quat_to_rot(pose.ego_q).T @ (p - np.reshape(pose.ego_t, (3, 1)))


def ego_to_global(p: np.ndarray, pose: SensorPose) -> np.ndarray:
    return geometry.quat_to_rot(pose.ego_q) @ p + np.reshape(pose.ego_t, (3, 1))


def quat_yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def load_points(path, dims: int = 5) -> np.ndarray:
    """nuScenes .pcd.bin: float32 records of `dims` values (x, y, z, intensity, ring)."""
    raw = np.fromfile(path, dtype=np.float32)
    return raw.reshape(-1, dims)


def project(points_cam: np.ndarray, K: np.ndarray, width: int, height: int, min_depth: float = 0.5):
    """Pixel coords (2xM), depths (M) and the index mask of points visible in the image."""
    front = points_cam[2] > min_depth
    uv = np.full((2, points_cam.shape[1]), -1.0)
    uv[:, front] = geometry.project_to_image(points_cam[:, front], K)
    inside = front & (uv[0] >= 0) & (uv[0] < width) & (uv[1] >= 0) & (uv[1] < height)
    return uv, points_cam[2], inside


def frame_from_rows(cam_row, lidar_row, raw_dir) -> FusionFrame:
    """FusionFrame from two rows of silver/frames.parquet (same sample)."""
    K = np.array([list(r) for r in cam_row.camera_intrinsic], dtype=np.float64)
    return FusionFrame(
        K=K,
        width=int(cam_row.width),
        height=int(cam_row.height),
        cam=SensorPose.from_row(cam_row),
        lidar=SensorPose.from_row(lidar_row),
        points_lidar=load_points(raw_dir / lidar_row.filename),
    )
