"""GATE stage: decide whether a new model may be promoted.

Rules (params.yaml -> gate):
  1. test mAP50 >= gate.min_map50
  2. no slice in the regression set / regression tags drops by more than
     gate.max_slice_drop compared with the baseline (the current production
     model's eval.json, e.g. from the main branch)

CI runs this on every pull request; a failing gate blocks the merge.
"""

from __future__ import annotations

import json
from pathlib import Path

from avdata import paths
from avdata.config import Params
from avdata.utils import get_logger, write_json

log = get_logger(__name__)


def run(p: Params, baseline: Path | None = None) -> dict:
    ev = json.loads((paths.METRICS / "eval.json").read_text())
    checks = []
    m = ev["overall"]["map50"]
    checks.append(
        {
            "rule": "min_map50",
            "value": m,
            "threshold": p.gate.min_map50,
            "passed": m >= p.gate.min_map50,
        }
    )

    if baseline and Path(baseline).exists():
        base = json.loads(Path(baseline).read_text())
        watched = ["regression_set", *p.curation.regression_tags]
        for name in watched:
            new, old = ev["slices"].get(name), base.get("slices", {}).get(name)
            if not new or not old:
                continue
            drop = old["map50"] - new["map50"]
            checks.append(
                {
                    "rule": f"slice_drop:{name}",
                    "value": round(drop, 4),
                    "threshold": p.gate.max_slice_drop,
                    "passed": drop <= p.gate.max_slice_drop,
                }
            )
    else:
        log.info("no baseline given - only absolute thresholds are checked")

    result = {
        "passed": all(c["passed"] for c in checks),
        "model_version": ev["model_version"],
        "checks": checks,
    }
    write_json(result, paths.METRICS / "gate.json")
    for c in checks:
        log.info(
            "  %-28s %s (value=%s, threshold=%s)",
            c["rule"],
            "PASS" if c["passed"] else "FAIL",
            c["value"],
            c["threshold"],
        )
    return result
