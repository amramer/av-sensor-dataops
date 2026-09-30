"""EVALUATE stage: test-set metrics overall and per scenario slice.

A single mAP hides where a perception model fails. Slices come from the
same scenario tags used for coverage KPIs (night, rain, location, speed,
has_<class>, ...) plus the regression set, so a weak slice points directly
at the data that needs to be collected or labeled next.
"""

from __future__ import annotations

from collections import defaultdict

from avdata import paths
from avdata.config import Params
from avdata.train import common
from avdata.utils import get_logger, read_parquet, write_json

log = get_logger(__name__)
MIN_SLICE_IMAGES = 1


def _metrics(res) -> dict[str, float]:
    return {
        "map50": round(float(res.box.map50), 4),
        "map50_95": round(float(res.box.map), 4),
        "precision": round(float(res.box.mp), 4),
        "recall": round(float(res.box.mr), 4),
    }


def run(p: Params) -> dict:
    from ultralytics import YOLO

    model_path = paths.MODELS / "detector.pt"
    model = YOLO(str(model_path))
    data_yaml = common.dataset_view()
    kw = {
        "imgsz": p.train.imgsz,
        "batch": p.train.batch,
        "device": common.resolve_device(p.train.device),
        "verbose": False,
        "plots": False,
        "project": str(common.WORK.resolve()),
        "name": "eval",
        "exist_ok": True,
    }

    overall = _metrics(model.val(data=str(data_yaml), split="test", **kw))
    index = read_parquet(paths.YOLO_DIR / "index.parquet")
    test = index[index["split"] == "test"]

    groups: dict[str, list[str]] = defaultdict(list)
    for r in test.itertuples():
        for tag in r.tags:
            groups[tag].append(r.frame_token)
        if r.in_regression_set:
            groups["regression_set"].append(r.frame_token)

    slices = {"test": overall | {"images": len(test)}}
    for name, tokens in sorted(groups.items()):
        if len(tokens) < MIN_SLICE_IMAGES:
            continue
        res = model.val(data=str(common.slice_yaml(name, tokens)), split="val", **kw)
        slices[name] = _metrics(res) | {"images": len(tokens)}
        log.info("  slice %-24s n=%-4d mAP50=%.3f", name, len(tokens), slices[name]["map50"])

    import hashlib

    result = {
        "model_version": hashlib.sha256(model_path.read_bytes()).hexdigest()[:12],
        "overall": overall,
        "slices": slices,
    }
    write_json(result, paths.METRICS / "eval.json")
    log.info("test mAP50=%.3f mAP50-95=%.3f", overall["map50"], overall["map50_95"])
    return result
