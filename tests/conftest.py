"""Shared fixtures: a synthetic nuScenes-format dataset in a temporary workspace.

Every stage writes relative to the working directory, so each test session
runs inside its own tmp dir with its own params.yaml.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import yaml

from avdata.config import load_params
from avdata.fixture import make_fixture

REPO = Path(__file__).resolve().parents[1]


def _workspace(tmp: Path, faults: bool, **overrides) -> Path:
    make_fixture(tmp / "raw", samples_per_scene=5, faults=faults)
    params = yaml.safe_load((REPO / "params.yaml").read_text())
    params["data"]["raw_dir"] = str(tmp / "raw")
    params["etl"]["workers"] = 1
    for section, values in overrides.items():
        params[section].update(values)
    (tmp / "params.yaml").write_text(yaml.safe_dump(params))
    return tmp


@pytest.fixture(scope="session")
def clean_ws(tmp_path_factory):
    """Workspace with a clean dataset, data stages already run up to curation."""
    from avdata.curate import export_yolo
    from avdata.curate import run as curate
    from avdata.etl import extract, features, transform
    from avdata.quality import run as quality

    ws = _workspace(tmp_path_factory.mktemp("clean"), faults=False)
    old = os.getcwd()
    os.chdir(ws)
    p = load_params(ws / "params.yaml")
    for stage in (extract, transform, features, quality, curate, export_yolo):
        stage.run(p)
    yield ws, p
    os.chdir(old)


@pytest.fixture()
def faulty_ws(tmp_path):
    ws = _workspace(tmp_path, faults=True)
    old = os.getcwd()
    os.chdir(ws)
    yield ws, load_params(ws / "params.yaml")
    os.chdir(old)
    shutil.rmtree(ws, ignore_errors=True)


@pytest.fixture()
def in_clean(clean_ws):
    ws, p = clean_ws
    old = os.getcwd()
    os.chdir(ws)
    yield ws, p
    os.chdir(old)
