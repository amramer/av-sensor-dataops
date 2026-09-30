"""Container entry point for hosted demos (Render, Hugging Face, any PaaS).

The model and demo samples are DVC-tracked, so they are not in the Git
checkout a PaaS builds from. On start-up this fetches the published bundle
(`scripts/package_demo.py` -> GitHub release asset) if the model is missing,
then starts Uvicorn on $PORT. A new model is deployed by publishing a new
bundle and restarting the service; no image rebuild needed.

    BUNDLE_URL=https://github.com/<you>/<repo>/releases/download/demo-latest/demo-bundle.tar.gz \\
    python -m avdata.serve.bootstrap
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import urllib.request
from pathlib import Path


def fetch_bundle(url: str, dest: Path) -> None:
    print(f"downloading demo bundle from {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():  # refuse paths escaping the destination
            if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts:
                raise RuntimeError(f"unsafe path in bundle: {m.name}")
        tar.extractall(dest)
    print(f"bundle extracted to {dest} ({len(data) / 1e6:.1f} MB)", flush=True)


def main() -> None:
    root = Path(os.environ.get("APP_ROOT", "."))
    model = root / "models" / "detector.onnx"
    url = os.environ.get("BUNDLE_URL")
    if not model.exists():
        if not url:
            sys.exit("models/detector.onnx missing and BUNDLE_URL not set")
        fetch_bundle(url, root)
    os.chdir(root)
    import uvicorn

    uvicorn.run(
        "avdata.serve.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        workers=int(os.environ.get("WEB_CONCURRENCY", "1")),
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
