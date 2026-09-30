"""Train on Kaggle's free GPU from the command line (used by Airflow when TRAIN_BACKEND=kaggle).

1. pushes notebooks/kaggle/train_kaggle.py as a private GPU kernel pinned to the
   current Git commit (so the kernel trains exactly this code + dataset version)
2. waits for the kernel to finish
3. downloads models/detector.pt and metrics/train.json

Afterwards `dvc commit -f train` records the outputs in dvc.lock, so lineage is
the same as for a local run.

Needs: `pip install kaggle`, ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY),
the repo pushed to GitHub, dvc.lock committed and `dvc push` done.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

SLUG = os.environ.get("KAGGLE_KERNEL_SLUG", "avdata-train")
POLL_S = 60


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def main() -> None:
    user = (
        os.environ.get("KAGGLE_USERNAME")
        or json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    )
    repo = os.environ.get("AVDATA_REPO_URL") or run(["git", "remote", "get-url", "origin"]).strip()
    commit = run(["git", "rev-parse", "HEAD"]).strip()
    if run(["git", "status", "--porcelain", "--", "src", "dvc.lock", "params.yaml"]).strip():
        raise SystemExit("commit (and push) your changes first: Kaggle trains the pushed commit")

    work = Path("runs/kaggle_kernel")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    code = Path("notebooks/kaggle/train_kaggle.py").read_text()
    (work / "train_kaggle.py").write_text(
        code.replace("__REPO_URL__", repo).replace("__COMMIT__", commit)
    )
    kernel_id = f"{user}/{SLUG}"
    (work / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel_id,
                "title": SLUG,
                "code_file": "train_kaggle.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        )
    )

    print(run(["kaggle", "kernels", "push", "-p", str(work)]))
    while True:
        time.sleep(POLL_S)
        status = run(["kaggle", "kernels", "status", kernel_id]).lower()
        print(status.strip())
        if "complete" in status:
            break
        if "error" in status or "cancel" in status:
            raise SystemExit(f"Kaggle kernel failed: {status}")

    out = Path("runs/kaggle_output")
    shutil.rmtree(out, ignore_errors=True)
    run(["kaggle", "kernels", "output", kernel_id, "-p", str(out)])
    Path("models").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)
    shutil.copy(next(out.rglob("detector.pt")), "models/detector.pt")
    shutil.copy(next(out.rglob("train.json")), "metrics/train.json")
    print("model downloaded from Kaggle - run `dvc commit -f train` to record it in dvc.lock")


if __name__ == "__main__":
    main()
