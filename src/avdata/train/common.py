"""Helpers shared by training, evaluation and export.

Ultralytics writes `labels/<split>.cache` next to the label folders. If it
did that inside data/ml_ready (a DVC output), DVC would see the dataset as
modified. So every run works on a view: `runs/dataset/` holds symlinks to the
DVC-tracked files, and the caches land in `runs/` instead.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np
import yaml

from avdata import paths

WORK = Path("runs")
VIEW = WORK / "dataset"
SPLITS = ("train", "val", "test")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def dataset_view() -> Path:
    """Build runs/dataset (symlinks to the DVC dataset) and return its data.yaml."""
    src = paths.YOLO_DIR.resolve()
    if not (src / "data.yaml").exists():
        raise FileNotFoundError(
            f"{src}/data.yaml missing - run `dvc repro export_yolo` or `dvc pull`"
        )
    if VIEW.exists():
        shutil.rmtree(VIEW)
    # per-file links inside real folders: Ultralytics resolves folder paths, so a
    # folder symlink would lead it straight back into the DVC output
    for kind in ("images", "labels"):
        for split in SPLITS:
            (VIEW / kind / split).mkdir(parents=True)
            for f in (src / kind / split).iterdir():
                link = VIEW / kind / split / f.name
                try:
                    link.symlink_to(f)
                except OSError:  # filesystems without symlink support
                    shutil.copy2(f, link)
    cfg = yaml.safe_load((src / "data.yaml").read_text())
    cfg["path"] = str(VIEW.resolve())
    out = VIEW / "data.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def slice_yaml(name: str, frame_tokens: list[str], split: str = "test") -> Path:
    """data.yaml whose val set is a subset of images (one scenario slice)."""
    base = yaml.safe_load((VIEW / "data.yaml").read_text())
    lst = WORK / "slices" / f"{name}.txt"
    lst.parent.mkdir(parents=True, exist_ok=True)
    # resolve only the view root, never the split symlinks (see module docstring)
    root = VIEW.resolve()
    lst.write_text(
        "\n".join(str(root / "images" / split / f"{t}.jpg") for t in frame_tokens) + "\n"
    )
    cfg = {
        "path": base["path"],
        "train": base["train"],
        "val": str(lst.resolve()),
        "names": base["names"],
    }
    out = WORK / "slices" / f"{name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def resolve_device(device: str) -> str | None:
    return None if device == "auto" else device
