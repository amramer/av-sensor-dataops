"""TRAIN stage: fine-tune a YOLO detector on the ML-ready dataset.

Traceability: every MLflow run is tagged with the git commit and the DVC
hash of the exact dataset version, so any model can be traced back to the
data, code and params that produced it (and reproduced with
`git checkout <commit> && dvc checkout && dvc repro train`).

Runs anywhere the repo can be cloned: laptop CPU (smoke test), Colab or
Kaggle GPU (see notebooks/train_colab.ipynb), or a Slurm/Kubernetes GPU job.
"""

from __future__ import annotations

import json
import os
import shutil

from avdata import paths
from avdata.config import Params
from avdata.train import common
from avdata.utils import dvc_out_hash, get_logger, git_commit, write_json

log = get_logger(__name__)
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "av-camera-detector")


def lineage_tags() -> dict[str, str]:
    return {
        "git_commit": git_commit(),
        "dataset_version": dvc_out_hash("data/ml_ready") or "untracked",
        "gold_version": dvc_out_hash("data/gold") or "untracked",
    }


def run(p: Params) -> dict:
    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})  # we log to MLflow ourselves, with lineage tags
    t = p.train
    common.seed_everything(t.seed)
    data_yaml = common.dataset_view()
    manifest = json.loads((paths.YOLO_DIR / "manifest.json").read_text())

    mlflow = _mlflow()
    tags = lineage_tags()
    run_ctx = mlflow.start_run(tags=tags) if mlflow else _NullCtx()
    with run_ctx:
        if mlflow:
            mlflow.log_params(
                t.model_dump() | {"n_train_images": manifest["splits"]["train"]["images"]}
            )
            mlflow.log_dict(manifest, "dataset_manifest.json")
        model = YOLO(t.model)
        model.train(
            data=str(data_yaml),
            epochs=t.epochs,
            imgsz=t.imgsz,
            batch=t.batch,
            seed=t.seed,
            deterministic=True,
            device=common.resolve_device(t.device),
            workers=t.workers,
            fraction=t.fraction,
            project=str(common.WORK.resolve()),
            name="train",
            exist_ok=True,
            verbose=False,
            plots=True,
        )
        best = common.WORK / "train" / "weights" / "best.pt"
        paths.MODELS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, paths.MODELS / "detector.pt")

        rd = {
            k.replace("(B)", "").replace("metrics/", ""): float(v)
            for k, v in (model.trainer.metrics or {}).items()
        }
        metrics = {"val": rd, "epochs": t.epochs, **tags}
        write_json(metrics, paths.METRICS / "train.json")
        if mlflow:
            mlflow.log_metrics({k.replace("-", "_"): v for k, v in rd.items()})
            mlflow.log_artifact(str(paths.MODELS / "detector.pt"), artifact_path="model")
            for plot in ("results.png", "confusion_matrix.png"):
                if (common.WORK / "train" / plot).exists():
                    mlflow.log_artifact(str(common.WORK / "train" / plot), artifact_path="plots")
    log.info("training done: %s", metrics["val"])
    return metrics


def _mlflow():
    """MLflow module, or None when it's not installed (tracking is optional)."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed - skipping experiment tracking")
        return None
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        # local default: SQLite store (MLflow 3 no longer accepts ./mlruns as a backend)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)
    log.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    return mlflow


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
