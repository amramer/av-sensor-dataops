"""Package the promoted model + demo samples into one downloadable bundle.

    python scripts/package_demo.py            # -> dist/demo-bundle.tar.gz

Contents: models/detector.onnx, models/model_card.json, models/fusion.json,
demo/ (camera images, LiDAR sweeps, calibration, ground truth). The hosted
demo downloads this at start-up (see src/avdata/serve/bootstrap.py).
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

FILES = ["models/detector.onnx", "models/model_card.json", "models/fusion.json"]


def main() -> None:
    missing = [f for f in [*FILES, "demo/index.json"] if not Path(f).exists()]
    if missing:
        raise SystemExit(
            f"missing {missing}: run `dvc repro demo_bundle fuse3d` or `dvc pull` first"
        )
    out = Path("dist/demo-bundle.tar.gz")
    out.parent.mkdir(exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for f in FILES:
            tar.add(f)
        tar.add("demo")
    card = json.loads(Path("models/model_card.json").read_text())
    n = len(json.loads(Path("demo/index.json").read_text()))
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB, model {card['model_version']}, {n} samples")


if __name__ == "__main__":
    main()
