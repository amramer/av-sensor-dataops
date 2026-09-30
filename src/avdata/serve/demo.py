"""Demo bundle: a handful of real test samples packaged with the model.

A deployed demo cannot reach the dataset, so `avdata demo-bundle` exports
N test samples (camera image, LiDAR sweep, calibration, ground truth) into
`demo/`. The API loads it at start-up and serves full camera + LiDAR
inference on those samples; any client can also POST its own frame to
`/predict3d` with the same calibration format.

demo/
  index.json               list of samples (id, scene, description, tags, n_gt)
  <id>/camera.jpg
  <id>/lidar.bin           float32 N x 4 (x, y, z, intensity) in the LiDAR frame
  <id>/meta.json           calibration (FusionFrame.calib_dict) + ground truth
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avdata.fusion.frames import FusionFrame, SensorPose

DEMO_DIR = Path("demo")


@dataclass
class DemoSample:
    id: str
    meta: dict
    camera_path: Path
    lidar_path: Path

    def frame(self) -> FusionFrame:
        return frame_from_calib(
            self.meta["calib"], np.fromfile(self.lidar_path, np.float32).reshape(-1, 4)
        )


def frame_from_calib(calib: dict, points: np.ndarray) -> FusionFrame:
    """FusionFrame from the JSON calibration format used by the API and the demo bundle."""
    return FusionFrame(
        K=np.asarray(calib["camera_intrinsic"], dtype=np.float64),
        width=int(calib["width"]),
        height=int(calib["height"]),
        cam=SensorPose.from_dict(calib["camera"]),
        lidar=SensorPose.from_dict(calib["lidar"]),
        points_lidar=np.asarray(points, dtype=np.float32),
    )


def load(demo_dir: Path = DEMO_DIR) -> dict[str, DemoSample]:
    idx = demo_dir / "index.json"
    if not idx.exists():
        return {}
    out = {}
    for s in json.loads(idx.read_text()):
        d = demo_dir / s["id"]
        out[s["id"]] = DemoSample(
            s["id"],
            json.loads((d / "meta.json").read_text()) | s,
            d / "camera.jpg",
            d / "lidar.bin",
        )
    return out


def build(p, n: int | None = None, out: Path = DEMO_DIR) -> dict:
    """Export test samples for the demo (called by `avdata demo-bundle`)."""
    from avdata.fusion import run as fusion
    from avdata.utils import read_parquet

    pairs = fusion.test_frames(p)
    anns = read_parquet(fusion.paths.ANNOTATIONS)
    by_sample = {k: g for k, g in anns.groupby("sample_token")}
    n = min(n or p.viz.demo_samples, len(pairs))
    pick = np.unique(np.linspace(0, len(pairs) - 1, n).round().astype(int)) if n else []
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    index = []
    first_ts = pairs.groupby("scene_name")["timestamp"].min()
    for i in pick:
        r = pairs.iloc[i]
        fr = fusion.build_frame(r, p.data.raw_dir)
        sid = f"{r.scene_name}-{int(r.timestamp) // 1000}"
        d = out / sid
        d.mkdir()
        shutil.copy2(p.data.raw_dir / r.filename, d / "camera.jpg")
        fr.points_lidar[:, :4].astype(np.float32).tofile(d / "lidar.bin")
        gt = fusion.gt_boxes(
            by_sample.get(r.sample_token, anns.iloc[:0]), fr, r.sample_token, p.fusion.class_range_m
        )
        meta = {
            "calib": fr.calib_dict(),
            "gt": [{k: g[k] for k in ("cls", "center", "size", "yaw", "distance")} for g in gt],
        }
        (d / "meta.json").write_text(json.dumps(meta))
        tags = [t for t in (r.tags if isinstance(r.tags, (list, np.ndarray)) else [])]
        index.append(
            {
                "id": sid,
                "scene": r.scene_name,
                "description": r.scene_description,
                "location": r.location,
                "tags": [str(t) for t in tags],
                "n_gt": len(gt),
                "t": round((int(r.timestamp) - int(first_ts[r.scene_name])) / 1e6, 1),
            }
        )
    (out / "index.json").write_text(json.dumps(index, indent=1))
    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    return {"samples": len(index), "size_mb": round(size_mb, 2), "dir": str(out)}
