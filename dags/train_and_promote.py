"""Train, evaluate, gate and promote the camera detector.

Triggered automatically whenever `sensor_data_pipeline` publishes a new
ML-ready dataset (Airflow asset), or manually.

  train     local GPU/CPU (`dvc repro train`), or on Kaggle's free GPU when
            TRAIN_BACKEND=kaggle (scripts/train_on_kaggle.py pushes a kernel,
            waits for it and brings the weights back)
  evaluate  test set + every scenario slice + regression set
  gate      fails the run if the model misses the bar or regresses on a slice
  export    ONNX + model card
  fuse3d    camera + LiDAR 3D detection on the test split, 3D metrics
  viz       visual outputs (renders, scene animations, failure gallery)
  demo      demo bundle for the hosted API
  promote   registers the model in MLflow and moves the `champion` alias
  release   commits + tags `model-*`; with AVDATA_GIT_PUSH=1 the tag triggers the
            release-demo workflow (live demo + results site)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import Asset, chain, dag

# the helper module lives next to the DAG files
sys.path.insert(0, str(Path(__file__).parent))
from avdata_tasks import dvc_stage, stage_task

ML_READY = Asset("avdata://datasets/ml_ready")
TRAIN_BACKEND = os.environ.get("TRAIN_BACKEND", "local")


@dag(
    dag_id="train_and_promote",
    schedule=[ML_READY],
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5), "owner": "ml-eng"},
    tags=["avdata", "training", "mlops"],
    doc_md=__doc__,
)
def train_and_promote():
    if TRAIN_BACKEND == "kaggle":
        train = stage_task(
            "train",
            "python scripts/train_on_kaggle.py && dvc commit -f train",
            execution_timeout=timedelta(hours=10),
        )
    else:
        train = dvc_stage("train", execution_timeout=timedelta(hours=10))
    evaluate = dvc_stage("evaluate")
    gate = stage_task(
        "quality_gate", "avdata gate --baseline models/production_eval.json", retries=0
    )
    export = dvc_stage("export_onnx")
    fuse3d = dvc_stage("fuse3d")
    viz = dvc_stage("viz")
    demo = dvc_stage("demo_bundle")
    promote = stage_task("promote", "avdata promote")
    publish = stage_task("publish_model_metrics", "avdata publish-kpis")
    release = stage_task("commit_model_release", "bash scripts/commit_release.sh model")

    chain(train, evaluate, gate, export, fuse3d, [viz, demo], promote, [publish, release])


train_and_promote()
