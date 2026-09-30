"""Object size priors (width, length, height in metres) per detection class.

The fusion uses a prior size for each class: a LiDAR cluster seen from one
side does not reveal the full extent of an object. Defaults are approximate
nuScenes class averages; `from_annotations` replaces them with medians
measured on the training split, so priors always match the data they serve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PRIORS: dict[str, tuple[float, float, float]] = {
    "car": (1.95, 4.62, 1.73),
    "truck": (2.52, 6.94, 2.84),
    "bus": (2.94, 11.19, 3.47),
    "trailer": (2.92, 12.28, 3.87),
    "construction_vehicle": (2.73, 6.37, 3.19),
    "motorcycle": (0.77, 2.11, 1.47),
    "bicycle": (0.61, 1.70, 1.30),
    "pedestrian": (0.67, 0.73, 1.77),
    "traffic_cone": (0.41, 0.41, 1.07),
    "barrier": (2.49, 0.48, 0.99),
}


def from_annotations(annotations: pd.DataFrame, min_count: int = 5) -> dict[str, dict]:
    """Median (w, l, h) per class from annotations; falls back to defaults when rare."""
    out: dict[str, dict] = {}
    known = annotations[annotations["det_class"].notna()]
    for cls, default in DEFAULT_PRIORS.items():
        rows = known[known["det_class"] == cls]
        if len(rows) >= min_count:
            size = np.median(np.vstack(rows["size"].to_numpy()), axis=0)
            out[cls] = {
                "size": [round(float(v), 3) for v in size],
                "n": len(rows),
                "source": "data",
            }
        else:
            out[cls] = {"size": list(default), "n": len(rows), "source": "default"}
    return out


def as_sizes(priors: dict) -> dict[str, tuple[float, float, float]]:
    return {
        k: tuple(v["size"]) if isinstance(v, dict) else tuple(v)
        for k, v in (priors or DEFAULT_PRIORS).items()
    }
