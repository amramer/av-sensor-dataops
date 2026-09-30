"""Command line entry point. Every DVC stage and Airflow task calls one command.

avdata extract | transform | features | quality | curate | export-yolo | kpis
avdata train | evaluate | gate | export-onnx
avdata fixture <dir>       # synthetic nuScenes-format data for tests/CI
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from avdata.config import load_params

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
ParamsOpt = Annotated[Path, typer.Option("--params", "-p", help="params.yaml to use")]


def _echo(result: object) -> None:
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def extract(params: ParamsOpt = Path("params.yaml")) -> None:
    """Raw nuScenes JSON tables -> bronze Parquet."""
    from avdata.etl import extract as stage

    _echo(stage.run(load_params(params)))


@app.command()
def transform(params: ParamsOpt = Path("params.yaml")) -> None:
    """Bronze -> silver frames, annotations, sync tables."""
    from avdata.etl import transform as stage

    _echo(stage.run(load_params(params)))


@app.command()
def features(params: ParamsOpt = Path("params.yaml")) -> None:
    """Decode every sensor file and compute per-frame stats."""
    from avdata.etl import features as stage

    _echo(stage.run(load_params(params)))


@app.command()
def quality(params: ParamsOpt = Path("params.yaml")) -> None:
    """Schema + sensor + label checks. Fails on ERROR issues."""
    from avdata.quality import run as stage

    try:
        _echo(stage.run(load_params(params)))
    except stage.QualityGateError as e:
        typer.secho(f"QUALITY GATE FAILED: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from e


@app.command()
def curate(params: ParamsOpt = Path("params.yaml")) -> None:
    """Scenario tags, scene-level splits, regression set, 2D labels."""
    from avdata.curate import run as stage

    _echo(stage.run(load_params(params)))


@app.command("export-yolo")
def export_yolo(params: ParamsOpt = Path("params.yaml")) -> None:
    """Gold -> YOLO dataset (images, labels, data.yaml, slice lists)."""
    from avdata.curate import export_yolo as stage

    _echo(stage.run(load_params(params)))


@app.command()
def kpis(params: ParamsOpt = Path("params.yaml")) -> None:
    """Spark job: data KPIs, coverage matrix, KPI history table, coverage report."""
    from avdata.kpi import run as stage

    _echo(stage.run(load_params(params)))


@app.command("publish-kpis")
def publish_kpis(params: ParamsOpt = Path("params.yaml")) -> None:
    """Append this run's KPIs, issues, coverage (and model metrics) to the KPI store."""
    from avdata.kpi import store

    _echo(store.run(load_params(params)))


@app.command()
def train(params: ParamsOpt = Path("params.yaml")) -> None:
    """Fine-tune the detector, log to MLflow."""
    from avdata.train import train as stage

    _echo(stage.run(load_params(params)))


@app.command()
def evaluate(params: ParamsOpt = Path("params.yaml")) -> None:
    """Evaluate on test + every scenario slice."""
    from avdata.train import evaluate as stage

    _echo(stage.run(load_params(params)))


@app.command()
def gate(
    params: ParamsOpt = Path("params.yaml"),
    baseline: Annotated[Path | None, typer.Option(help="metrics.json of the current model")] = None,
) -> None:
    """Block promotion when the model misses the bar or regresses on a slice."""
    from avdata.train import gate as stage

    result = stage.run(load_params(params), baseline)
    _echo(result)
    if not result["passed"]:
        raise typer.Exit(code=3)


@app.command()
def promote(params: ParamsOpt = Path("params.yaml")) -> None:
    """Register the gated ONNX model in MLflow and move the `champion` alias."""
    from avdata.train import promote as stage

    try:
        _echo(stage.run(load_params(params)))
    except stage.PromotionError as e:
        typer.secho(f"NOT PROMOTED: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=4) from e


@app.command("export-onnx")
def export_onnx(params: ParamsOpt = Path("params.yaml")) -> None:
    """Export the trained model to ONNX + write the model card."""
    from avdata.train import export as stage

    _echo(stage.run(load_params(params)))


@app.command()
def benchmark(
    runs: int = 100,
    warmup: int = 10,
    providers: Annotated[
        list[str] | None, typer.Option(help="ONNX Runtime providers, in order")
    ] = None,
) -> None:
    """Latency/FPS of the ONNX model on this machine (laptop, GPU, Jetson)."""
    from avdata.serve import benchmark as bench

    _echo(bench.run(runs=runs, warmup=warmup, providers=providers))


@app.command()
def fixture(
    out: Path,
    samples_per_scene: int = 5,
    faults: Annotated[bool, typer.Option(help="inject data problems")] = False,
) -> None:
    """Write a small synthetic dataset in nuScenes format."""
    from avdata.fixture import make_fixture

    make_fixture(out, samples_per_scene=samples_per_scene, faults=faults)
    typer.echo(f"fixture written to {out}")


if __name__ == "__main__":
    app()
