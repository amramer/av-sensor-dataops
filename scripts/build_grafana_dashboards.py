"""Generate the Grafana dashboards (JSON) from compact Python definitions.

    python scripts/build_grafana_dashboards.py

Writing dashboards as code keeps them reviewable in pull requests; the
generated JSON in monitoring/grafana/dashboards/ is what Grafana loads.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "monitoring" / "grafana" / "dashboards"
PG = {"type": "grafana-postgresql-datasource", "uid": "kpi-postgres"}
PROM = {"type": "prometheus", "uid": "prometheus"}
BLUE, ORANGE = "#2a78d6", "#eb6834"

LATEST_RUN = "(SELECT run_id FROM kpi_history ORDER BY published_at DESC LIMIT 1)"


def latest(metric: str) -> str:
    return (
        f"SELECT value FROM kpi_history WHERE metric = '{metric}' "
        "ORDER BY published_at DESC LIMIT 1"
    )


def sql_target(sql: str, fmt: str = "table") -> dict:
    return {
        "datasource": PG,
        "rawSql": sql,
        "format": fmt,
        "rawQuery": True,
        "editorMode": "code",
        "refId": "A",
    }


def prom_target(expr: str, legend: str, ref: str = "A") -> dict:
    return {"datasource": PROM, "expr": expr, "legendFormat": legend, "refId": ref}


def panel(
    kind: str,
    title: str,
    targets: list[dict],
    x: int,
    y: int,
    w: int,
    h: int,
    unit: str | None = None,
    color: str | None = BLUE,
    desc: str = "",
    **opts,
) -> dict:
    defaults: dict = {"unit": unit} if unit else {}
    defaults["color"] = (
        {"mode": "fixed", "fixedColor": color} if color else {"mode": "palette-classic"}
    )
    if kind == "timeseries":
        defaults["custom"] = {
            "lineWidth": 2,
            "fillOpacity": 0,
            "showPoints": "always",
            "pointSize": 8,
        }
    return {
        "type": kind,
        "title": title,
        "description": desc,
        "targets": targets,
        "datasource": targets[0]["datasource"],
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": opts or {},
    }


def stat(title: str, metric: str, x: int, unit: str | None = None, desc: str = "") -> dict:
    return panel(
        "stat",
        title,
        [sql_target(latest(metric))],
        x,
        0,
        6,
        4,
        unit=unit,
        desc=desc,
        colorMode="none",
        graphMode="none",
        textMode="value",
    )


def dashboard(uid: str, title: str, panels: list[dict], refresh: str = "1m") -> dict:
    for i, p in enumerate(panels, start=1):
        p["id"] = i
    return {
        "uid": uid,
        "title": title,
        "tags": ["avdata"],
        "timezone": "utc",
        "schemaVersion": 39,
        "version": 1,
        "refresh": refresh,
        "time": {"from": "now-30d", "to": "now"},
        "panels": panels,
        "templating": {"list": []},
        "annotations": {"list": []},
    }


def series(prefix: str) -> str:
    """One time series per metric under a prefix (the suffix becomes the series name)."""
    return (
        f"SELECT published_at AS time, replace(metric, '{prefix}', '') AS metric, value "
        f"FROM kpi_history WHERE metric LIKE '{prefix}%' ORDER BY 1"
    )


def latest_group(prefix: str, label: str) -> str:
    """Latest run's values of all metrics under a prefix, as (label, value) rows."""
    return (
        f"SELECT replace(metric, '{prefix}', '') AS {label}, value FROM kpi_history "
        f"WHERE run_id = {LATEST_RUN} AND metric LIKE '{prefix}%' ORDER BY value DESC"
    )


def data_dashboard() -> dict:
    return dashboard(
        "avdata-data",
        "Sensor data KPIs",
        [
            stat(
                "Frames",
                "volume.frames_total",
                0,
                desc="All sensor frames ingested (keyframes + sweeps)",
            ),
            stat("Data size", "volume.bytes_total", 6, unit="decbytes"),
            stat(
                "ML-ready samples",
                "readiness.ml_ready_rate",
                12,
                unit="percentunit",
                desc="Keyframe samples whose camera frame passed QC and sync checks",
            ),
            stat(
                "Labels OK",
                "labels.ok_rate",
                18,
                unit="percentunit",
                desc="3D boxes that are neither weak (no lidar points) nor poorly visible",
            ),
            panel(
                "timeseries",
                "Data growth (frames per pipeline run)",
                [
                    sql_target(
                        "SELECT published_at AS time, value AS frames FROM kpi_history "
                        "WHERE metric = 'volume.frames_total' ORDER BY 1",
                        "time_series",
                    )
                ],
                0,
                4,
                12,
                8,
            ),
            panel(
                "timeseries",
                "Frame QC pass rate by channel",
                [sql_target(series("quality.frame_pass_rate_by_channel."), "time_series")],
                12,
                4,
                12,
                8,
                unit="percentunit",
                color=None,
            ),
            panel(
                "barchart",
                "Quality issues by check (latest run)",
                [sql_target(latest_group("quality.issues.", "check_name"))],
                0,
                12,
                12,
                8,
                orientation="horizontal",
                xField="check_name",
            ),
            panel(
                "table",
                "Scenario coverage (latest run)",
                [
                    sql_target(
                        "SELECT location, time_of_day, weather, SUM(samples) AS samples, "
                        "SUM(ml_ready) AS ml_ready FROM coverage WHERE run_id = "
                        "(SELECT run_id FROM coverage ORDER BY published_at DESC LIMIT 1) "
                        "GROUP BY 1, 2, 3 ORDER BY samples"
                    )
                ],
                12,
                12,
                12,
                8,
                desc="Sorted ascending: the top rows are the conditions with the least data",
            ),
            panel(
                "barchart",
                "Training boxes per class (latest run)",
                [sql_target(latest_group("readiness.train_boxes_by_class.", "class"))],
                0,
                20,
                12,
                8,
                orientation="horizontal",
                xField="class",
            ),
            panel(
                "barchart",
                "Model mAP50 per scenario slice (latest model)",
                [
                    sql_target(
                        "SELECT slice, value AS map50 FROM model_metrics WHERE metric = 'map50' "
                        "AND run_id = (SELECT run_id FROM model_metrics ORDER BY published_at "
                        "DESC LIMIT 1) ORDER BY value"
                    )
                ],
                12,
                20,
                12,
                8,
                unit="percentunit",
                orientation="horizontal",
                xField="slice",
            ),
            panel(
                "timeseries",
                "Test mAP50 across model versions",
                [
                    sql_target(
                        'SELECT published_at AS time, value AS "test mAP50" FROM model_metrics '
                        "WHERE slice = 'test' AND metric = 'map50' ORDER BY 1",
                        "time_series",
                    )
                ],
                0,
                28,
                24,
                7,
                unit="percentunit",
            ),
            panel(
                "barchart",
                "3D detection (camera + LiDAR): AP per class, latest model",
                [
                    sql_target(
                        "SELECT replace(slice, '3d_detector:', '') AS class_name, value AS ap "
                        "FROM model_metrics WHERE metric = 'ap' AND slice LIKE '3d_detector:%' "
                        "AND run_id = (SELECT run_id FROM model_metrics WHERE slice LIKE '3d_%' "
                        "ORDER BY published_at DESC LIMIT 1) ORDER BY value"
                    )
                ],
                0,
                35,
                12,
                8,
                unit="percentunit",
                orientation="horizontal",
                xField="class_name",
                desc="nuScenes-style AP (centre distance 0.5-4 m), 2D detections lifted with LiDAR",
            ),
            panel(
                "timeseries",
                "3D mAP and NDS-lite across model versions",
                [
                    sql_target(
                        "SELECT published_at AS time, slice || ' ' || metric AS metric, value "
                        "FROM model_metrics WHERE slice IN ('3d_detector', '3d_oracle_2d') "
                        "AND metric IN ('mAP', 'NDS_lite') ORDER BY 1",
                        "time_series",
                    )
                ],
                12,
                35,
                12,
                8,
                unit="percentunit",
                color=None,
            ),
        ],
    )


def api_dashboard() -> dict:
    rate = "[5m]"
    return dashboard(
        "avdata-api",
        "Inference service",
        [
            panel(
                "timeseries",
                "Requests per second by status",
                [prom_target(f"sum by (status) (rate(avdata_requests_total{rate}))", "{{status}}")],
                0,
                0,
                12,
                8,
                unit="reqps",
                color=None,
            ),
            panel(
                "timeseries",
                "Model latency",
                [
                    prom_target(
                        "histogram_quantile(0.5, sum by (le) "
                        f"(rate(avdata_inference_seconds_bucket{rate})))",
                        "p50",
                    ),
                    prom_target(
                        "histogram_quantile(0.95, sum by (le) "
                        f"(rate(avdata_inference_seconds_bucket{rate})))",
                        "p95",
                        "B",
                    ),
                ],
                12,
                0,
                12,
                8,
                unit="s",
                color=None,
            ),
            panel(
                "timeseries",
                "Detections per second by class (output drift)",
                [prom_target(f"sum by (cls) (rate(avdata_detections_total{rate}))", "{{cls}}")],
                0,
                8,
                12,
                8,
                color=None,
            ),
            panel(
                "timeseries",
                "Median input brightness (input drift: night vs day traffic)",
                [
                    prom_target(
                        "histogram_quantile(0.5, sum by (le) "
                        "(rate(avdata_input_brightness_bucket[15m])))",
                        "median grey level",
                    )
                ],
                12,
                8,
                12,
                8,
                color=ORANGE,
                desc="Training data brightness comes from the KPI report; a sustained drop means "
                "the service sees more night images than the model was trained on.",
            ),
        ],
        refresh="10s",
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for d in (data_dashboard(), api_dashboard()):
        (OUT / f"{d['uid']}.json").write_text(json.dumps(d, indent=2) + "\n")
        print("wrote", OUT / f"{d['uid']}.json")
