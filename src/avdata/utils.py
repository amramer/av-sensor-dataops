"""Small shared helpers: logging, Parquet I/O, JSON reports, lineage info."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.environ.get("AVDATA_LOG_LEVEL", "INFO"), format=LOG_FORMAT)
    return logging.getLogger(name)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the upstream stage first (`dvc repro`).")
    return pd.read_parquet(path)


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (Path, datetime)):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str:
    """Current commit (+ '-dirty' when there are uncommitted changes)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], stderr=subprocess.DEVNULL
        ).returncode
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA", "unknown")


def dvc_out_hash(path: str, lock_file: Path = Path("dvc.lock")) -> str | None:
    """md5 that dvc.lock records for an output path (the dataset version id)."""
    if not lock_file.exists():
        return None
    import yaml

    lock = yaml.safe_load(lock_file.read_text()) or {}
    for stage in (lock.get("stages") or {}).values():
        for out in stage.get("outs", []):
            if out.get("path") == path:
                return out.get("md5")
    return None
