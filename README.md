# av-sensor-dataops

**Automated pipeline that turns raw autonomous-vehicle sensor logs into versioned, quality-checked, ML-ready datasets, then trains, gates, serves and monitors a camera detector on them.**

Built on [nuScenes](https://www.nuscenes.org/nuscenes) (camera, LiDAR, radar, GPS/IMU recordings from real test vehicles), starting with the front camera and the top LiDAR. More sensors are a config change, not a code change.

```
raw sensor logs ─► bronze ─► silver ─► quality gate ─► gold ─► ML-ready ─► train ─► evaluate ─► gate ─► ONNX ─► API
   (DVC)          Parquet    joined,     schema +        tags,    YOLO       MLflow   per-scenario  regression  model   FastAPI
                             synced,     sensor + label  splits,  format              slices        checks      card    + Prometheus
                             features    checks          labels
                                  └──────────────► Spark KPIs ─► KPI store ─► Grafana
```

| | |
|---|---|
| **Orchestration** | Airflow 3 (data-aware scheduling: a new dataset version triggers training) |
| **Versioning & lineage** | Git + DVC (`dvc.lock` hashes every input/output), MLflow runs tagged with commit + dataset hash, Git tags per data/model release |
| **Processing** | pandas / PyArrow, PySpark (parallel decoding, KPI aggregation) |
| **Data quality** | Pandera schemas + sensor checks (sync, dropped frames, corrupt files, degraded LiDAR) + label checks (weak labels, visibility) |
| **ML** | YOLOv8 fine-tuning, per-scenario evaluation, regression set, promotion gate, MLflow model registry |
| **Serving** | ONNX Runtime + FastAPI, Prometheus metrics, latency benchmark (CPU / GPU / Jetson) |
| **Monitoring** | Grafana: data volume & growth, quality, scenario coverage, label quality, ML readiness, model and service metrics |
| **Infra** | Docker Compose (MinIO, Postgres, MLflow, Airflow, Prometheus, Grafana), Kubernetes manifests, GitHub Actions |

---

## Quick start (no download needed)

```bash
git clone https://github.com/<you>/av-sensor-dataops && cd av-sensor-dataops
python -m venv .venv && source .venv/bin/activate
make install                 # needs Java 17+ for PySpark

make fixture                 # synthetic data in the exact nuScenes format (3 scenes, ~18 MB)
dvc repro export_yolo kpis   # raw -> ML-ready dataset + KPI report
open reports/coverage.html   # KPI snapshot
dvc dag                      # the pipeline graph
```

## With real nuScenes data

1. Register at [nuscenes.org](https://www.nuscenes.org/nuscenes#download), download **nuScenes mini** (v1.0-mini, about 4 GB, 10 scenes) and extract it to `data/raw/nuscenes/` (you should see `v1.0-mini/`, `samples/`, `sweeps/`, `maps/`).
2. Track it and run:

```bash
dvc add data/raw/nuscenes          # raw data is now versioned; the .dvc file goes into Git
git add data/raw/nuscenes.dvc data/raw/.gitignore && git commit -m "raw: nuScenes mini"
dvc repro export_yolo kpis         # data pipeline on your laptop (CPU is enough)
```

3. Train on a free GPU (Colab or Kaggle, see [Training on a free GPU](#training-on-a-free-gpu)), or locally with `dvc repro`.

To scale up, set `data.version: v1.0-trainval` and add trainval parts to `data/raw/nuscenes/`; nothing else changes.

## Data flow in detail

| Stage | Command | Output | What happens |
|---|---|---|---|
| extract | `avdata extract` | `data/bronze/*.parquet` | 13 nuScenes JSON tables → typed Parquet, 1:1. Manifest with row counts + SHA-256 of each source file |
| transform | `avdata transform` | `data/silver/frames, annotations, sync` | One row per sensor frame: calibration, ego pose, scene, file status, inter-frame timing, ego speed. Keyframe offset of every sensor to the reference sensor |
| features | `avdata features` | `data/silver/frame_features` | Decodes **every** file once (pandas process pool or Spark): brightness/contrast/sharpness for images, point count/range/intensity for LiDAR, point count for radar. Decode failures are recorded |
| quality | `avdata quality` | `reports/quality/*`, `frames_qc`, `annotations_qc` | Schemas + rule checks (below). **Fails the pipeline on ERRORs** |
| curate | `avdata curate` | `data/gold/samples, labels_2d` | Scenario tags, scene-level stratified splits (no leakage), regression set, 3D→2D box projection, ML-readiness per sample |
| export_yolo | `avdata export-yolo` | `data/ml_ready/yolo/` | Images + YOLO labels + `data.yaml` + index of tags per image |
| kpis | `avdata kpis` (Spark) | `reports/kpis.json`, `coverage.html` | Volume, measured sensor rates, sync percentiles, QC pass rates, label quality, coverage matrix with empty cells, class balance |
| train | `avdata train` | `models/detector.pt` | YOLOv8 fine-tune; MLflow run tagged with git commit + dataset hash |
| evaluate | `avdata evaluate` | `metrics/eval.json` | mAP overall and per scenario slice (night, rain, location, speed, has_bicycle, …) + regression set |
| export_onnx | `avdata export-onnx` | `models/detector.onnx`, `model_card.json` | ONNX + model card (classes, input spec, eval, lineage) |

Outside DVC (they have side effects): `avdata gate` (blocks promotion), `avdata promote` (MLflow registry, `champion` alias), `avdata publish-kpis` (append to the KPI store), `avdata benchmark`.

### Quality checks

| Check | Severity | Catches |
|---|---|---|
| schema | ERROR | wrong types, nulls, broken keys, impossible values (Pandera) |
| missing_file | ERROR | metadata points to a file that is not on disk |
| corrupt_file | ERROR | truncated LiDAR sweep, undecodable or wrong-size image, bad PCD |
| duplicate_frame | ERROR | two frames on one channel with the same timestamp |
| sync_offset / missing_keyframe | WARN | sensor keyframe more than `sync_tolerance_ms` from the reference sensor |
| frame_gap | WARN | dropped frames (gap > 2.5 × nominal period) |
| low_lidar_points | WARN | blocked or degraded LiDAR sweep |
| night_metadata_mismatch | WARN | scene says "night" but images are bright (metadata error) |
| weak_label | WARN | box with no LiDAR points inside (unverifiable ground truth) |

Frames with ERRORs or sync problems are excluded from the ML-ready set, and the reason is kept per sample. `tests/test_etl_quality.py` injects each fault into a dataset and checks that it is caught.

### Adding a sensor

```yaml
# params.yaml
sensors:
  enabled: [CAM_FRONT, LIDAR_TOP, RADAR_FRONT]
```

`dvc repro` re-runs from `transform`: radar frames get ingested, decoded, rate- and sync-checked and show up in the KPIs. A new modality (e.g. thermal) is one entry in `src/avdata/sensors.py`. `tests/test_etl_quality.py::test_adding_a_sensor_is_config_only` proves it.

## Reproducibility and lineage

- **Everything is a DVC stage.** `dvc.lock` records the hash of every dependency, parameter and output of the last run. `git checkout <tag> && dvc pull` restores any dataset or model version exactly.
- **Deterministic outputs.** No timestamps or random ids inside stage outputs; fixed seeds for splits and training. CI checks that `dvc status` is clean right after a full run.
- **Traceable models.** MLflow runs and the model card carry `git_commit`, `dataset_version` (hash of `data/ml_ready`) and `gold_version`.
- **Releases.** The Airflow DAGs commit `dvc.lock` + reports and tag `data-YYYYMMDD-HHMM` / `model-…`.

```bash
dvc params diff        # what changed in the settings
dvc metrics diff main  # how quality, KPIs and model metrics moved vs main
dvc exp run -S train.epochs=100 -S curation.min_box_px=10   # tracked experiment
```

## Local platform (Docker Compose)

```bash
make up            # MinIO :9001, Postgres, MLflow :5000, Prometheus :9090, Grafana :3000
make airflow       # + Airflow :8080 (admin password is printed in the container logs)
make api           # + inference API :8000/docs (after models/detector.onnx exists)
```

Use MinIO as the DVC remote and the MLflow server:

```bash
dvc remote add -d minio s3://dvc-store && dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify --local minio access_key_id avdata
dvc remote modify --local minio secret_access_key avdata-secret
export MLFLOW_TRACKING_URI=http://localhost:5000 AWS_ACCESS_KEY_ID=avdata \
       AWS_SECRET_ACCESS_KEY=avdata-secret MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AVDATA_KPI_DB_URL=postgresql+psycopg://avdata:avdata@localhost:5432/kpi
dvc push && avdata publish-kpis
```

**Airflow DAGs**

- `sensor_data_pipeline` (daily): waits for a new drive in `data/landing/`, ingests it, runs every stage as its own task, pushes to the DVC remote, publishes KPIs, commits and tags the release. Emits the `ml_ready` asset.
- `train_and_promote` (triggered by `ml_ready`): train → evaluate → gate → export → promote → publish.

Simulate a fleet: ingest 4 of the 10 mini scenes, then feed the rest as "new drives" and watch the growth in Grafana:

```bash
python scripts/ingest_drive.py --set scene-0061 scene-0103 scene-0553 scene-0655
python scripts/ingest_drive.py --stage scene-0757 scene-0796 scene-0916   # a new drive lands
# trigger sensor_data_pipeline in the Airflow UI
```

## Training on a free GPU

- **Colab:** open `notebooks/train_colab.ipynb`. It clones the repo, `dvc pull`s exactly the dataset version in `dvc.lock`, runs `dvc repro export_onnx` and `dvc push`es the model. No Google Drive involved.
- **Kaggle (automated):** `python scripts/train_on_kaggle.py` pushes a private GPU kernel pinned to the current commit, waits, and downloads the weights; `dvc commit -f train` records them. Airflow uses it with `TRAIN_BACKEND=kaggle`.

Free hosted options that fit: Cloudflare R2 (S3 API, no egress fees) or DagsHub as the DVC remote, and DagsHub's hosted MLflow for tracking. Setup: [docs/setup.md](docs/setup.md).

## Serving and edge benchmarking

```bash
uvicorn avdata.serve.app:app --port 8000
curl -F image=@some_frame.jpg localhost:8000/predict
curl localhost:8000/metrics/          # latency histogram, detections per class, input brightness
avdata benchmark --runs 200           # p50/p95/p99 per stage -> reports/benchmarks/<host>_<provider>.json
```

The API exports input brightness and the predicted class mix, so a shift (e.g. far more night traffic than in the training data) is visible in Grafana before accuracy complaints arrive. For NVIDIA Jetson (TensorRT FP16/INT8) see [docs/jetson.md](docs/jetson.md).

## Kubernetes

```bash
make kind                              # local cluster, API with 2 replicas + HPA + probes
kubectl apply -k deploy/k8s            # API + workspace volume + nightly pipeline CronJob
```

With Airflow on Kubernetes (official Helm chart), set `AVDATA_RUNNER=k8s`: every DAG task becomes a pod from the pipeline image.

## CI/CD

- **ci.yml** (every push/PR): lint → tests (incl. Spark and API) → full `dvc repro` on synthetic data with CPU training → gate → reproducibility check → API container smoke test. The pipeline image is pushed to GHCR on `main`.
- **data-pipeline.yml** (weekly / manual): runs the data stages on the real data from the DVC remote and opens a pull request with the new `dvc.lock` and reports, so dataset releases are reviewed like code.

## Repository layout

```
src/avdata/
  config.py  paths.py  sensors.py  geometry.py  fixture.py  cli.py
  etl/        extract.py transform.py features.py
  quality/    schemas.py checks.py run.py
  curate/     scenarios.py labels2d.py run.py export_yolo.py
  kpi/        run.py (Spark) report.py (Plotly) store.py (SQL)
  train/      train.py evaluate.py gate.py export.py promote.py common.py
  serve/      app.py inference.py benchmark.py
dags/                   Airflow DAGs
docker/                 images: pipeline, api, airflow, mlflow
monitoring/             Prometheus + Grafana (dashboards generated by scripts/)
deploy/k8s/             Kubernetes manifests + kind config
notebooks/              Colab + Kaggle training
tests/                  unit + integration tests on synthetic nuScenes-format data
dvc.yaml  params.yaml   the pipeline and its settings
```

## Roadmap

- LiDAR stage: point cloud → BEV/range images, LiDAR-camera projection check as a calibration QC
- Model-vs-label disagreement mining to flag suspicious ground truth for re-labeling
- Drift report comparing live inputs (Prometheus) with training-set statistics
- OpenLineage events from Airflow for dataset-level lineage in Marquez
- TensorRT INT8 calibration and Jetson Orin benchmark results

## License

Code: MIT. nuScenes data is © Motional and licensed separately (CC BY-NC-SA 4.0, non-commercial use).
