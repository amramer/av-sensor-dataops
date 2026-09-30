"""PROMOTE: make a gated model the production ("champion") model.

1. refuses unless metrics/gate.json says the gate passed for this exact model
2. logs the ONNX model with its model card and eval report to MLflow,
   registers it as a new version of `av-camera-detector` and moves the
   `champion` alias to it (the registry is the source of truth for deployment)
3. stores the eval report as models/production_eval.json, the baseline the
   next candidate's gate compares against
"""

from __future__ import annotations

import json
import os
import shutil

from avdata import paths
from avdata.config import Params
from avdata.train.train import EXPERIMENT, lineage_tags
from avdata.utils import get_logger

log = get_logger(__name__)
REGISTERED_NAME = os.environ.get("MLFLOW_MODEL_NAME", "av-camera-detector")


class PromotionError(RuntimeError):
    pass


def run(p: Params) -> dict:
    gate_file = paths.METRICS / "gate.json"
    if not gate_file.exists():
        raise PromotionError("metrics/gate.json missing - run `avdata gate` first")
    gate = json.loads(gate_file.read_text())
    card = json.loads((paths.MODELS / "model_card.json").read_text())
    ev = json.loads((paths.METRICS / "eval.json").read_text())
    if gate["model_version"] != ev["model_version"]:
        raise PromotionError("gate.json belongs to a different model - re-run evaluate and gate")
    if not gate["passed"]:
        raise PromotionError(f"gate failed for model {gate['model_version']} - not promoting")

    result = {"model_version": card["model_version"], "registered": None}
    try:
        import mlflow
        import onnx
        from mlflow import MlflowClient
    except ImportError:
        log.warning("mlflow/onnx not installed - skipping model registry")
    else:
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"promote-{card['model_version']}", tags=lineage_tags()):
            mlflow.log_dict(card, "model_card.json")
            mlflow.log_dict(ev, "eval.json")
            mlflow.log_metrics({f"test_{k}": v for k, v in ev["overall"].items()})
            info = mlflow.onnx.log_model(
                onnx.load(str(paths.MODELS / "detector.onnx")),
                name="model",
                registered_model_name=REGISTERED_NAME,
            )
        client = MlflowClient()
        version = info.registered_model_version
        client.set_registered_model_alias(REGISTERED_NAME, "champion", version)
        for k in ("model_version", "git_commit", "dataset_version"):
            client.set_model_version_tag(REGISTERED_NAME, version, k, str(card.get(k)))
        result["registered"] = {"name": REGISTERED_NAME, "version": version, "alias": "champion"}
        log.info("registered %s v%s as champion", REGISTERED_NAME, version)

    shutil.copy2(paths.METRICS / "eval.json", paths.MODELS / "production_eval.json")
    return result
