# Setup guide

Step by step, from an empty laptop to the full platform. Everything here is free.

## 1. Local environment

```bash
python -m venv .venv && source .venv/bin/activate
make install            # all extras + pre-commit hooks
java -version           # PySpark needs Java 17+ (macOS: brew install openjdk@17)
pytest -m "not train"   # ~1 minute, no downloads
```

## 2. Get nuScenes mini

1. Create an account at <https://www.nuscenes.org/sign-up>, accept the terms.
2. Download page → *Full dataset (v1.0)* → **Mini** (`v1.0-mini.tgz`, about 4 GB).
3. Extract into the repo:

```bash
mkdir -p data/raw/nuscenes
tar -xzf v1.0-mini.tgz -C data/raw/nuscenes
ls data/raw/nuscenes        # maps  samples  sweeps  v1.0-mini
dvc add data/raw/nuscenes
git add data/raw/nuscenes.dvc data/raw/.gitignore && git commit -m "raw: nuScenes mini"
```

## 3. A DVC remote you can reach from Colab and Kaggle

Google Drive mounts are not versioned and CI cannot reach them. Use object storage instead.

**Option A: Cloudflare R2** (S3 API, free tier, no egress fees)

1. Cloudflare dashboard → R2 → create bucket `av-dataops`.
2. R2 → Manage API tokens → create a token with *Object Read & Write* on that bucket. Note the
   access key id, secret and your account id.

```bash
dvc remote add -d r2 s3://av-dataops/dvc
dvc remote modify r2 endpointurl https://<ACCOUNT_ID>.r2.cloudflarestorage.com
dvc remote modify r2 region auto
dvc remote modify --local r2 access_key_id <KEY_ID>          # --local: stays out of Git
dvc remote modify --local r2 secret_access_key <SECRET>
dvc push
```

The code is identical for AWS S3 (drop the endpoint) or Azure Blob (`dvc remote add -d az azure://container/path`).

**Option B: DagsHub** (Git mirror + DVC storage + hosted MLflow in one place). Connect your GitHub
repo on dagshub.com; its *Remote* button shows the exact DVC and MLflow URLs and credentials.

## 4. Experiment tracking

- Local: nothing to do, runs go to `mlflow.db` (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
- Compose: `make up` and `export MLFLOW_TRACKING_URI=http://localhost:5000`.
- Hosted (reachable from Colab/Kaggle): DagsHub MLflow: set `MLFLOW_TRACKING_URI`,
  `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`.

## 5. Train on a free GPU

**Colab:** push the repo to GitHub, open `notebooks/train_colab.ipynb` from GitHub in Colab, add the
secrets it lists (key icon on the left), select a GPU runtime and run all cells.

**Kaggle:**

```bash
pip install kaggle             # API token: kaggle.com -> Settings -> Create New Token -> ~/.kaggle/kaggle.json
# in a Kaggle notebook: Add-ons -> Secrets -> add DVC_ACCESS_KEY_ID, DVC_SECRET_ACCESS_KEY (+ MLflow ones)
git push && dvc push           # Kaggle trains the pushed commit and data
python scripts/train_on_kaggle.py
dvc commit -f train && dvc repro evaluate export_onnx
```

## 6. Dashboards

```bash
make up
export AVDATA_KPI_DB_URL=postgresql+psycopg://avdata:avdata@localhost:5432/kpi
avdata publish-kpis            # after each pipeline run (Airflow does this for you)
```

Grafana <http://localhost:3000> (admin/admin) → folder *AV data platform*:
*Sensor data KPIs* (volume, growth, QC pass rate per channel, issues, coverage, class balance,
per-scenario mAP) and *Inference service* (throughput, latency, class mix, input brightness).

## 7. Airflow

```bash
make airflow                   # http://localhost:8080, user admin, password in `docker compose logs airflow`
python scripts/ingest_drive.py --set scene-0061 scene-0103 scene-0553 scene-0655
python scripts/ingest_drive.py --stage scene-0757 scene-0796
```

Trigger `sensor_data_pipeline` in the UI. When it finishes, `train_and_promote` starts by itself
(asset-triggered). Set `TRAIN_BACKEND=kaggle` in the compose file to train on Kaggle from there.
