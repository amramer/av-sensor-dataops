"""Spark KPI job and KPI store (needs a Java runtime for PySpark)."""

import json
import shutil

import pytest
from sqlalchemy import create_engine, text

from avdata import paths

pytestmark = pytest.mark.spark

requires_java = pytest.mark.skipif(shutil.which("java") is None, reason="PySpark needs Java")


@requires_java
def test_kpis_and_publish(in_clean, monkeypatch):
    from avdata.kpi import run as kpis
    from avdata.kpi import store

    ws, p = in_clean
    result = kpis.run(p)
    assert result["volume"]["frames_total"] > 0
    assert set(result["volume"]["by_channel"]) == {"CAM_FRONT", "LIDAR_TOP"}
    assert result["readiness"]["ml_ready_rate"] == 1.0
    assert result["coverage"]["cells_observed"] == 3
    assert "night/rain" in " ".join(result["coverage"]["empty_cells"])
    assert (paths.REPORTS / "coverage.html").stat().st_size > 1000
    json.loads((paths.REPORTS / "kpis.json").read_text())

    db = ws / "kpi_test.db"
    monkeypatch.setenv("AVDATA_KPI_DB_URL", f"sqlite:///{db}")
    store.run(p)
    store.run(p)  # second run appends a new snapshot (history for Grafana)
    with create_engine(f"sqlite:///{db}").connect() as c:
        runs = c.execute(text("SELECT COUNT(DISTINCT run_id) FROM kpi_history")).scalar()
    assert runs == 2
