# Common tasks. `make help` lists them.
.DEFAULT_GOAL := help
PY ?= python

help:  ## show this help
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## dev install (all extras, CPU PyTorch via ultralytics)
	pip install -e ".[spark,serve,train,dvc,postgres,dev]"
	pre-commit install

fixture:  ## synthetic nuScenes-format data in data/raw/nuscenes (no download needed)
	avdata fixture data/raw/nuscenes

data:  ## run the data pipeline up to the ML-ready dataset + KPIs
	dvc repro export_yolo kpis

pipeline:  ## run everything, including training, evaluation and ONNX export
	dvc repro

test:  ## all tests except the slow training smoke test
	pytest -m "not train"

test-all:  ## all tests
	pytest

lint:  ## ruff lint + format check
	ruff check . && ruff format --check .

format:  ## auto-format
	ruff check --fix . && ruff format .

up:  ## start MinIO, Postgres, MLflow, Prometheus, Grafana
	docker compose up -d

airflow:  ## start the platform + Airflow (http://localhost:8080)
	docker compose --profile airflow up -d --build

api:  ## start the inference API (needs models/detector.onnx)
	docker compose --profile api up -d --build api

down:  ## stop all containers
	docker compose --profile airflow --profile api down

publish:  ## push KPIs of the current run to the KPI store (Grafana)
	avdata publish-kpis

dashboards:  ## regenerate Grafana dashboard JSON
	$(PY) scripts/build_grafana_dashboards.py

benchmark:  ## ONNX latency on this machine
	avdata benchmark --runs 200

fusion:  ## camera + LiDAR 3D detection, visual outputs, demo bundle
	dvc repro fuse3d viz demo_bundle

serve:  ## API + demo UI at http://localhost:8000
	uvicorn avdata.serve.app:app --port 8000

site:  ## static results site in site/ (GitHub Pages) + demo bundle in dist/
	python scripts/package_demo.py
	python scripts/build_site.py

kind:  ## local Kubernetes cluster + API deployment
	kind create cluster --name avdata --config deploy/k8s/kind-cluster.yaml
	docker build -f docker/Dockerfile.serve -t ghcr.io/amramer/av-sensor-dataops-api:latest .
	kind load docker-image ghcr.io/amramer/av-sensor-dataops-api:latest --name avdata
	kubectl apply -f deploy/k8s/namespace.yaml -f deploy/k8s/api.yaml

.PHONY: help install fixture data pipeline test test-all lint format up airflow api down publish dashboards benchmark fusion serve site kind
