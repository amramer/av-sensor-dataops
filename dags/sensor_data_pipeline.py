"""Sensor data pipeline: new drive -> versioned, quality-checked ML-ready dataset.

Runs daily. It waits (in reschedule mode, so it holds no worker slot) for a
new drive in `data/landing/` marked by a `READY` file, ingests it, runs every
DVC stage as its own task (so failures retry and show up per stage), pushes
the new dataset version to the DVC remote, publishes KPIs for Grafana and
commits + tags the dataset release in Git.

When the ML-ready dataset is updated it emits the `ml_ready` asset, which
triggers the `train_and_promote` DAG (data-aware scheduling).

Local demo:  python scripts/ingest_drive.py --stage scene-0103 scene-0553
             (writes data/landing/scenes.txt + READY), then trigger the DAG.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import Asset, chain, dag, task

# the helper module lives next to the DAG files
sys.path.insert(0, str(Path(__file__).parent))
from avdata_tasks import PROJECT_DIR, dvc_stage, stage_task

ML_READY = Asset("avdata://datasets/ml_ready")


@dag(
    dag_id="sensor_data_pipeline",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2), "owner": "data-eng"},
    tags=["avdata", "etl", "data-quality"],
    doc_md=__doc__,
)
def sensor_data_pipeline():
    @task.sensor(poke_interval=300, timeout=6 * 3600, mode="reschedule", soft_fail=True, retries=0)
    def wait_for_new_drive() -> bool:
        """True once an upload is complete. No new drive today -> run is skipped, not failed."""
        return (Path(PROJECT_DIR) / "data" / "landing" / "READY").exists()

    wait = wait_for_new_drive()
    ingest = stage_task("ingest_new_drive", "python scripts/ingest_drive.py --apply")
    etl = [dvc_stage(s) for s in ("extract", "transform", "features")]
    quality = dvc_stage("quality", retries=0)  # a data problem will not fix itself on retry
    curate = dvc_stage("curate")
    export = dvc_stage("export_yolo", outlets=[ML_READY])
    kpis = dvc_stage("kpis")
    push = stage_task("dvc_push", "dvc push || echo 'no DVC remote configured - skipping push'")
    publish = stage_task("publish_kpis", "avdata publish-kpis")
    release = stage_task("commit_data_release", "bash scripts/commit_release.sh data")

    chain(wait, ingest, *etl, quality, curate, [export, kpis])
    kpis >> publish
    [export, publish] >> push >> release


sensor_data_pipeline()
