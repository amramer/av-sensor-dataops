"""Build the static results site (GitHub Pages) from a pipeline run.

    python scripts/build_site.py --demo-url https://av-sensor-fusion-demo.onrender.com

site/
  index.html        3D metrics, scene animations, samples, failure gallery (from reports/viz)
  coverage.html     dataset KPI report (Plotly)
  kpis.json  eval3d.json  quality.json   raw numbers for anyone who wants them

The site is static, so it is always online, unlike a free API instance that
sleeps when idle. It links to the live demo for interactive inference.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SITE = Path("site")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-url", default="", help="URL of the live API demo")
    ap.add_argument("--repo-url", default="", help="URL of the source repository")
    args = ap.parse_args()

    viz = Path("reports/viz")
    if not (viz / "index.html").exists():
        raise SystemExit("reports/viz missing: run `dvc repro viz` or `dvc pull reports/viz`")
    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(viz, SITE)
    for src, dst in (
        ("reports/coverage.html", "coverage.html"),
        ("reports/kpis.json", "kpis.json"),
        ("metrics/eval3d.json", "eval3d.json"),
        ("reports/quality/summary.json", "quality.json"),
    ):
        if Path(src).exists():
            shutil.copy2(src, SITE / dst)

    html = (
        (SITE / "index.html").read_text().replace('href="../coverage.html"', 'href="coverage.html"')
    )
    links = []
    if args.demo_url:
        links.append(f'<a href="{args.demo_url}">Live demo (API may take ~1 min to wake up)</a>')
    if args.repo_url:
        links.append(f'<a href="{args.repo_url}">Source code</a>')
    links.append('<a href="eval3d.json">3D metrics (JSON)</a>')
    bar = '<p style="display:flex;gap:18px;flex-wrap:wrap">' + " ".join(links) + "</p>"
    html = html.replace("<h2>3D metrics</h2>", bar + "<h2>3D metrics</h2>", 1)
    (SITE / "index.html").write_text(html)
    (SITE / ".nojekyll").touch()
    files = [p for p in SITE.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files) / 1e6
    print(json.dumps({"site": str(SITE), "files": len(files), "size_mb": round(size, 1)}))


if __name__ == "__main__":
    main()
