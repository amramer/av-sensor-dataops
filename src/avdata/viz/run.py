"""VIZ stage: the visual outputs of a pipeline run.

reports/viz/
  samples/<scene>_<n>.jpg     camera (LiDAR depth + 3D boxes) | bird's-eye view
  failures/<rank>_<scene>.jpg frames with the most missed objects (triage queue)
  <scene>.gif                 animation over every test keyframe of a scene
  lidar_camera/<scene>.jpg    LiDAR-on-camera overlay (calibration check at a glance)
  index.html                  gallery with 3D metrics, links to the KPI report
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from avdata import paths
from avdata.config import Params
from avdata.fusion import metrics
from avdata.fusion.run import PRED_DIR, build_frame, test_frames
from avdata.utils import get_logger, read_parquet, write_json
from avdata.viz import render

log = get_logger(__name__)
OUT = paths.REPORTS / "viz"


def _boxes(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "cls": r.cls,
            "center": list(r.center),
            "size": list(r.size),
            "yaw": float(r.yaw),
            "score": float(r.score),
            "source": r.source,
        }
        for r in df.itertuples()
    ]


def _gts(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "cls": r.cls,
            "center": list(r.center),
            "size": list(r.size),
            "yaw": float(r.yaw),
            "sample": r.sample,
        }
        for r in df.itertuples()
    ]


def missed_per_sample(preds: pd.DataFrame, gt: pd.DataFrame, thr: float) -> dict[str, int]:
    out = {}
    for s, g in gt.groupby("sample"):
        G = _gts(g)
        P = [dict(b, sample=s) for b in _boxes(preds[preds["sample_token"] == s])]
        hit = 0
        for cls in {x["cls"] for x in G}:
            _, _, pairs = metrics.match(
                [p for p in P if p["cls"] == cls], [x for x in G if x["cls"] == cls], thr
            )
            hit += len(pairs)
        out[s] = len(G) - hit
    return out


def render_sample(r, p: Params, preds: pd.DataFrame, gt: pd.DataFrame, subtitle: str = ""):
    fr = build_frame(r, p.data.raw_dir)
    image = Image.open(p.data.raw_dir / r.filename)
    g = _gts(gt[gt["sample"] == r.sample_token])
    b = _boxes(preds[preds["sample_token"] == r.sample_token])
    cam = render.render_camera(image, fr, g, b)
    bev = render.render_bev(fr, g, b, p.viz.bev_range_m)
    title = f"{r.scene_name} · {r.scene_description[:70]}"
    sub = subtitle or f"{len(b)} fused 3D boxes · {len(g)} ground-truth boxes in view"
    return render.compose(cam, bev, title, sub), fr, image


def run(p: Params) -> dict:
    boxes = read_parquet(PRED_DIR / "boxes3d.parquet")
    gt = read_parquet(PRED_DIR / "gt3d.parquet") if (PRED_DIR / "gt3d.parquet").exists() else None
    if gt is None or gt.empty:
        gt = pd.DataFrame(columns=["sample", "cls", "center", "size", "yaw"])
    mode = "detector" if (boxes["mode"] == "detector").any() else "oracle_2d"
    preds = boxes[boxes["mode"] == mode]
    pairs = test_frames(p)
    ev = json.loads((paths.METRICS / "eval3d.json").read_text())
    for sub in ("samples", "failures", "lidar_camera"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        for f in (OUT / sub).glob("*"):
            f.unlink()

    written = defaultdict(list)
    # 1. evenly spaced samples
    n = min(p.viz.samples, len(pairs))
    pick = np.unique(np.linspace(0, len(pairs) - 1, n).round().astype(int)) if n else []
    for i, idx in enumerate(pick):
        r = pairs.iloc[idx]
        img, _, _ = render_sample(r, p, preds, gt)
        path = OUT / "samples" / f"{r.scene_name}_{i:02d}.jpg"
        img.save(path, quality=88)
        written["samples"].append(path.name)

    # 2. failure gallery: most missed ground-truth boxes first
    missed = missed_per_sample(preds, gt, p.fusion.tp_threshold_m)
    worst = sorted((m, s) for s, m in missed.items() if m > 0)[::-1][: p.viz.failures]
    for rank, (m, s) in enumerate(worst, 1):
        r = pairs[pairs["sample_token"] == s].iloc[0]
        img, _, _ = render_sample(
            r,
            p,
            preds,
            gt,
            f"{m} ground-truth object(s) missed at {p.fusion.tp_threshold_m:g} m ({mode})",
        )
        path = OUT / "failures" / f"{rank:02d}_{r.scene_name}.jpg"
        img.save(path, quality=88)
        written["failures"].append(path.name)

    # 3. per-scene animation + LiDAR-on-camera overlay
    for scene, grp in pairs.groupby("scene_name"):
        frames = []
        for k, r in enumerate(grp.itertuples()):
            img, fr, image = render_sample(r, p, preds, gt)
            frames.append(img)
            if k == 0:
                overlay = render.render_camera(image, fr, lidar=True)
                overlay.save(OUT / "lidar_camera" / f"{scene}.jpg", quality=88)
                written["lidar_camera"].append(f"{scene}.jpg")
        render.save_gif(frames, OUT / f"{scene}.gif")
        written["gifs"].append(f"{scene}.gif")

    write_gallery(ev, mode, written)
    summary = {"mode": mode, **{k: len(v) for k, v in written.items()}}
    write_json(summary, OUT / "manifest.json")
    log.info("visual outputs: %s", summary)
    return summary


def write_gallery(ev: dict, mode: str, written: dict) -> None:
    def row(m):
        e = ev.get(m)
        if not e:
            return ""
        return (
            f"<tr><td>{m}</td><td>{e['mAP']:.3f}</td><td>{e['NDS_lite']:.3f}</td>"
            f"<td>{_f(e['mATE'])}</td><td>{_f(e['mASE'])}</td><td>{_f(e['mAOE'])}</td>"
            f"<td>{e['n_gt']}</td><td>{e['n_pred']}</td></tr>"
        )

    imgs = lambda sub, names: "".join(  # noqa: E731
        f'<figure><a href="{sub}/{n}"><img src="{sub}/{n}" loading="lazy"></a>'
        f"<figcaption>{n}</figcaption></figure>"
        for n in names
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Camera + LiDAR results</title>
<style>
:root{{--bg:#fcfcfb;--ink:#1b1b1a;--muted:#6b6a66;--line:#e2e0da;--blue:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#f2f2ef;--muted:#a9a8a2;
--line:#3a3a37}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px}} h1{{font-size:24px;margin:0 0 4px}}
p{{color:var(--muted);margin:0 0 16px}} h2{{font-size:17px;margin:28px 0 10px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
td,th{{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}}
figure{{margin:0}} img{{width:100%;border-radius:6px;display:block}}
figcaption{{font-size:12px;color:var(--muted)}} a{{color:var(--blue)}}
.wrap{{overflow-x:auto}}
</style></head><body><main>
<h1>Camera + LiDAR 3D detection</h1>
<p>Front camera detections lifted to 3D with LiDAR; evaluated on the test split with
nuScenes-style centre-distance matching. Aqua = ground truth, blue = fused (LiDAR),
orange = fused (monocular fallback). Gallery shows mode <b>{mode}</b>.
<a href="../coverage.html">Dataset KPI report</a></p>
<h2>3D metrics</h2><div class="wrap"><table><tr><th>2D source</th><th>mAP</th><th>NDS-lite</th>
<th>mATE (m)</th><th>mASE</th><th>mAOE (rad)</th><th>GT</th><th>pred</th></tr>
{row("oracle_2d")}{row("detector")}</table></div>
<h2>Scene animations</h2><div class="grid">{imgs(".", written.get("gifs", []))}</div>
<h2>Samples</h2><div class="grid">{imgs("samples", written.get("samples", []))}</div>
<h2>Failure gallery (most missed objects)</h2>
<div class="grid">{imgs("failures", written.get("failures", [])) or "<p>No misses.</p>"}</div>
<h2>LiDAR on camera (calibration at a glance)</h2>
<div class="grid">{imgs("lidar_camera", written.get("lidar_camera", []))}</div>
</main></body></html>"""
    (OUT / "index.html").write_text(html)


def _f(v):
    return "–" if v is None else f"{v:.3f}"
