"""Compact markdown summary of a pipeline run (for the GitHub Actions job summary)."""

import json
from pathlib import Path


def load(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


q = load("reports/quality/summary.json")
k = load("reports/kpis.json")
c = load("reports/curation.json")
e = load("metrics/eval.json")
e3 = load("metrics/eval3d.json")
g = load("metrics/gate.json")
cov = k.get("coverage", {})

rows = [
    ("Frames ingested", k.get("volume", {}).get("frames_total")),
    ("Data size (GB)", k.get("volume", {}).get("gb_total")),
    ("QC errors / warnings", f"{q.get('errors')} / {q.get('warnings')}"),
    ("Frame QC pass rate", q.get("frame_pass_rate")),
    ("Label OK rate", q.get("label_ok_rate")),
    ("ML-ready samples", f"{c.get('ml_ready_samples')} of {c.get('samples')}"),
    (
        "Samples train / val / test",
        " / ".join(str(c.get("samples_per_split", {}).get(s, 0)) for s in ("train", "val", "test")),
    ),
    (
        "Coverage cells observed",
        f"{cov.get('cells_observed')} of {cov.get('cells_possible')}",
    ),
    ("Test mAP50", e.get("overall", {}).get("map50")),
    ("Regression set mAP50", e.get("slices", {}).get("regression_set", {}).get("map50")),
    ("Gate", "passed" if g.get("passed") else "failed"),
    ("3D mAP, oracle 2D + LiDAR", e3.get("oracle_2d", {}).get("mAP")),
    ("3D mAP, detector + LiDAR", e3.get("detector", {}).get("mAP")),
    ("3D mATE (m), oracle 2D", e3.get("oracle_2d", {}).get("mATE")),
]
print("| Metric | Value |\n|---|---|")
for name, value in rows:
    print(f"| {name} | {value} |")
