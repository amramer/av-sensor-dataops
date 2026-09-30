"""Task factory shared by the DAGs (this file defines no DAG itself).

Every pipeline step is one shell command (`dvc repro --single-item <stage>`,
`avdata ...`). Where it runs is a deployment choice, set by AVDATA_RUNNER:

  bash  run in the Airflow worker, inside the project checkout (docker compose)
  k8s   run as a pod from the pipeline image (KubernetesPodOperator), with the
        project on a shared volume - how it runs on a cluster
"""

from __future__ import annotations

import os

PROJECT_DIR = os.environ.get("AVDATA_PROJECT_DIR", "/opt/project")
AVDATA_BIN = os.environ.get("AVDATA_BIN", "")  # venv with avdata + dvc, e.g. /opt/avdata/bin
RUNNER = os.environ.get("AVDATA_RUNNER", "bash")
IMAGE = os.environ.get("AVDATA_IMAGE", "ghcr.io/amramer/av-sensor-dataops-pipeline:latest")
PVC = os.environ.get("AVDATA_PVC", "avdata-workspace")
NAMESPACE = os.environ.get("AVDATA_NAMESPACE", "avdata")


def stage_task(task_id: str, command: str, **kwargs):
    """An operator that runs `command` from the project root."""
    if RUNNER == "k8s":
        from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
        from kubernetes.client import models as k8s

        return KubernetesPodOperator(
            task_id=task_id,
            name=f"avdata-{task_id.replace('_', '-')}",
            namespace=NAMESPACE,
            image=IMAGE,
            cmds=["sh", "-c"],
            arguments=[f"cd /workspace && {command}"],
            volumes=[
                k8s.V1Volume(
                    name="workspace",
                    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name=PVC),
                )
            ],
            volume_mounts=[k8s.V1VolumeMount(name="workspace", mount_path="/workspace")],
            env_from=[k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name="avdata-env"))],
            container_resources=k8s.V1ResourceRequirements(
                requests={"cpu": "1", "memory": "2Gi"}, limits={"cpu": "4", "memory": "8Gi"}
            ),
            get_logs=True,
            on_finish_action="delete_pod",
            **kwargs,
        )

    from airflow.providers.standard.operators.bash import BashOperator

    path = f"export PATH={AVDATA_BIN}:$PATH && " if AVDATA_BIN else ""
    return BashOperator(
        task_id=task_id,
        bash_command=f"{path}cd {PROJECT_DIR} && {command}",
        append_env=True,
        **kwargs,
    )


def dvc_stage(stage: str, **kwargs):
    return stage_task(stage, f"dvc repro --single-item {stage}", **kwargs)
