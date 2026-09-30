# Free deployment: live demo + results site

Two free, complementary targets:

| | What | Where | Cost | Availability |
|---|---|---|---|---|
| Live demo | FastAPI + demo UI: camera + LiDAR 3D inference on bundled samples, image upload, API docs | Render free web service (Docker) | free, no card | sleeps after 15 min idle, ~1 min to wake |
| Results site | 3D metrics, scene animations, sample renders, failure gallery, KPI report | GitHub Pages | free | always on |

The model and demo samples are DVC outputs, not Git files, so they travel as one bundle
(`dist/demo-bundle.tar.gz`, about 10-30 MB) published as the GitHub release `demo-latest`.
The Render container downloads it at start-up (`python -m avdata.serve.bootstrap`).

## 1. Produce the bundle and site locally (optional check)

```bash
dvc repro                        # includes fuse3d, viz, demo_bundle
python scripts/package_demo.py   # -> dist/demo-bundle.tar.gz
python scripts/build_site.py --demo-url https://<service>.onrender.com --repo-url https://github.com/<you>/av-sensor-dataops
python -m http.server -d site 8080   # preview the site
BUNDLE_URL=file-or-http-url APP_ROOT=/tmp/app python -m avdata.serve.bootstrap   # preview the hosted API
```

## 2. GitHub: Pages and secrets

1. Repository → Settings → Pages → Source: **GitHub Actions**.
2. Settings → Secrets and variables → Actions:
   - secrets `DVC_ACCESS_KEY_ID`, `DVC_SECRET_ACCESS_KEY` (use the real model from your DVC remote; without them the workflow runs the synthetic pipeline)
   - secret `RENDER_DEPLOY_HOOK` (step 3)
   - variable `DEMO_URL` = your Render URL (linked from the site)
3. Actions → **release-demo** → Run workflow. It publishes release `demo-latest` with the bundle and deploys the site to `https://<you>.github.io/av-sensor-dataops/`.

It also runs automatically on tags `model-*` (the Airflow `train_and_promote` DAG creates them).

## 3. Render: the live API

1. Sign up at render.com (GitHub login, no credit card).
2. New → **Blueprint** → select the repository. Render reads `render.yaml`
   (Docker, free plan, health check `/health`).
3. Set `BUNDLE_URL` to `https://github.com/<you>/av-sensor-dataops/releases/download/demo-latest/demo-bundle.tar.gz`.
4. Service → Settings → **Deploy Hook**: copy the URL into the GitHub secret `RENDER_DEPLOY_HOOK`.
5. Open `https://<service>.onrender.com`: the demo UI; `/docs` for the API.

Free instances have 512 MB RAM and a fraction of a CPU: the ONNX detector (YOLOv8n, 640 px)
takes roughly 0.5-2 s per image there. It is a demo, not a latency benchmark; use
`avdata benchmark` on real hardware for numbers.

## Alternatives

- **Hugging Face Spaces (Docker):** new accounts need PRO for Docker/Gradio Spaces on CPU since
  mid-2026. With PRO or a legacy free quota, create a Docker Space, push this repo, use
  `docker/Dockerfile.render` as `Dockerfile`, set `app_port: 10000` and the `BUNDLE_URL` secret.
- **Google Cloud Run:** generous free tier and scale-to-zero, but needs a billing account.
- **Your own machine:** `make api` (docker compose) or `uvicorn avdata.serve.app:app`.

## Data licence

nuScenes is CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike). The demo bundles a few
test frames for a non-commercial portfolio demo with attribution in the page footer. Check the
nuScenes terms of use before publishing frames, and keep the Space/site non-commercial.
