"""EXPORT: gold -> YOLO-format dataset (the ML-ready layer).

data/ml_ready/yolo/
  images/{train,val,test}/<frame_token>.jpg
  labels/{train,val,test}/<frame_token>.txt   class x_center y_center width height (normalised)
  data.yaml                                   class names + split folders
  index.parquet                               frame_token -> split, tags, regression flag

Only ML-ready samples are exported. Images are copied, not linked, so the
folder is self-contained for `dvc push` to Colab/Kaggle; DVC deduplicates
identical content in its cache, so this costs no extra remote storage.
"""

from __future__ import annotations

import shutil

import pandas as pd
import yaml

from avdata import paths
from avdata.config import Params
from avdata.utils import get_logger, read_parquet, write_json, write_parquet

log = get_logger(__name__)
SPLITS = ("train", "val", "test")


def yolo_line(r: pd.Series) -> str:
    xc = (r.x1 + r.x2) / 2 / r.img_w
    yc = (r.y1 + r.y2) / 2 / r.img_h
    w = (r.x2 - r.x1) / r.img_w
    h = (r.y2 - r.y1) / r.img_h
    return f"{int(r.class_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def run(p: Params) -> dict:
    samples = read_parquet(paths.SAMPLES)
    labels = read_parquet(paths.LABELS_2D)
    out = paths.YOLO_DIR
    if out.exists():
        shutil.rmtree(out)
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    ready = samples[samples["ml_ready"]]
    by_frame = {k: g for k, g in labels.groupby("frame_token")}
    counts = {s: {"images": 0, "boxes": 0} for s in SPLITS}
    for r in ready.itertuples():
        src = p.data.raw_dir / r.filename
        shutil.copy2(src, out / "images" / r.split / f"{r.frame_token}.jpg")
        boxes = by_frame.get(r.frame_token)
        lines = [yolo_line(b) for _, b in boxes.iterrows()] if boxes is not None else []
        (out / "labels" / r.split / f"{r.frame_token}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        counts[r.split]["images"] += 1
        counts[r.split]["boxes"] += len(lines)

    data_yaml = {
        "path": ".",  # resolved to an absolute path at train time
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(p.curation.class_names)),
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    index = ready[
        [
            "frame_token",
            "sample_token",
            "scene_name",
            "split",
            "tags",
            "in_regression_set",
            "time_of_day",
            "weather",
            "location",
            "speed_bin",
        ]
    ]
    write_parquet(index.reset_index(drop=True), out / "index.parquet")
    summary = {"splits": counts, "classes": p.curation.class_names}
    write_json(summary, out / "manifest.json")
    log.info("YOLO dataset: %s", counts)
    return summary
