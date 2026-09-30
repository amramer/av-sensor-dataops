"""Runs INSIDE a Kaggle kernel (free GPU). Pushed by scripts/train_on_kaggle.py.

Kaggle secrets used (Add-ons -> Secrets):
  DVC_ACCESS_KEY_ID, DVC_SECRET_ACCESS_KEY   credentials of the DVC remote (e.g. Cloudflare R2)
  MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD   (optional, e.g. DagsHub)
"""

import os
import shutil
import subprocess

REPO_URL = "__REPO_URL__"
COMMIT = "__COMMIT__"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


try:
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    for env, key in {
        "AWS_ACCESS_KEY_ID": "DVC_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY": "DVC_SECRET_ACCESS_KEY",
        "MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME": "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD": "MLFLOW_TRACKING_PASSWORD",
    }.items():
        try:
            os.environ[env] = secrets.get_secret(key)
        except Exception:
            print(f"secret {key} not set")
except ImportError:
    print("not running on Kaggle - using environment variables")

sh(f"git clone {REPO_URL} repo")
os.chdir("repo")
sh(f"git checkout {COMMIT}")
sh('pip install -q -e ".[train,dvc]"')
sh("dvc pull data/ml_ready")
sh("avdata train")
os.makedirs("/kaggle/working/out", exist_ok=True)
for f in ("models/detector.pt", "metrics/train.json"):
    shutil.copy(f, "/kaggle/working/out/")
print("done")
