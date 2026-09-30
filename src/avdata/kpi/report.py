"""Static HTML KPI report (Plotly), written next to kpis.json on every run.

The live dashboards are in Grafana; this file is the snapshot that travels
with a dataset version (it is a DVC output, so every data version has one).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE = "#2a78d6"  # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2
SEQUENTIAL = [[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#104281"]]  # one-hue ramp
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def write_html(kpis: dict, coverage: pd.DataFrame, out: Path) -> Path:
    fig = make_subplots(
        rows=2,
        cols=2,
        vertical_spacing=0.16,
        horizontal_spacing=0.14,
        subplot_titles=(
            "Scenario coverage (samples)",
            "Frames per sensor channel",
            "Quality issues by check",
            "Train 2D boxes per class",
        ),
    )

    # 1. coverage heatmap: location x (time of day / weather)
    cov = coverage.assign(condition=coverage["time_of_day"] + " / " + coverage["weather"])
    grid = cov.pivot_table(
        index="location", columns="condition", values="samples", aggfunc="sum", fill_value=0
    )
    for col in ["day / clear", "day / rain", "night / clear", "night / rain"]:
        if col not in grid:
            grid[col] = 0  # empty cells are the point: they show coverage gaps
    grid = grid[sorted(grid.columns)]
    fig.add_trace(
        go.Heatmap(
            z=grid.values,
            x=list(grid.columns),
            y=list(grid.index),
            colorscale=SEQUENTIAL,
            showscale=False,
            xgap=2,
            ygap=2,
            text=grid.values,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z} samples<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 2. frames per channel
    ch = kpis["volume"]["by_channel"]
    fig.add_trace(
        go.Bar(
            x=list(ch),
            y=[v["frames"] for v in ch.values()],
            marker_color=BLUE,
            name="frames",
            showlegend=False,
            customdata=[[v["mb"], v["measured_hz"]] for v in ch.values()],
            hovertemplate=(
                "%{x}<br>%{y} frames, %{customdata[0]} MB, %{customdata[1]} Hz<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    # 3. issues, split by severity (two series -> legend + two fixed slots)
    iss = pd.Series(kpis["quality"]["issues"], dtype=int)
    if iss.empty:
        iss = pd.Series({"none.WARN": 0})
    df = iss.rename_axis("key").reset_index(name="n")
    df[["check", "severity"]] = df["key"].str.rsplit(".", n=1, expand=True)
    for sev, color in (("ERROR", ORANGE), ("WARN", BLUE)):
        d = df[df["severity"] == sev]
        fig.add_trace(
            go.Bar(
                y=d["check"],
                x=d["n"],
                orientation="h",
                marker_color=color,
                name=sev,
                hovertemplate="%{y}: %{x}<extra>" + sev + "</extra>",
            ),
            row=2,
            col=1,
        )

    # 4. class balance of the training split
    tb = pd.Series(kpis["readiness"]["train_boxes_by_class"], dtype=int).sort_values()
    fig.add_trace(
        go.Bar(
            y=list(tb.index),
            x=list(tb.values),
            orientation="h",
            marker_color=BLUE,
            showlegend=False,
            hovertemplate="%{y}: %{x} boxes<extra></extra>",
        ),
        row=2,
        col=2,
    )

    missing = kpis["readiness"].get("train_classes_missing", [])
    title = (
        f"Dataset KPIs: {kpis['volume']['frames_total']} frames, "
        f"{kpis['volume']['gb_total']:.2f} GB, "
        f"ML-ready {100 * kpis['readiness']['ml_ready_rate']:.0f}%"
    )
    fig.update_layout(
        title={
            "text": title
            + (f"<br><sub>No training boxes for: {', '.join(missing)}</sub>" if missing else ""),
            "font": {"color": INK},
        },
        template="plotly_white",
        height=820,
        bargap=0.35,
        barmode="group",
        font={"family": "system-ui, sans-serif", "color": INK_2, "size": 13},
        legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
        margin={"t": 120, "l": 60, "r": 30, "b": 50},
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return out
