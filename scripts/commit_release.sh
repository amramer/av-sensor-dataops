#!/usr/bin/env bash
# Commit and tag a data or model release produced by the pipeline.
#   bash scripts/commit_release.sh data|model
# The commit holds only small, reviewable files (dvc.lock, *.dvc, params,
# metrics, reports); the data itself is in the DVC remote. The tag makes the
# release reproducible:  git checkout data-20261001-0930 && dvc pull
set -euo pipefail
kind="${1:?usage: commit_release.sh data|model}"
stamp="$(date -u +%Y%m%d-%H%M)"

git add dvc.lock params.yaml data/raw/*.dvc reports/*.json reports/quality/summary.json \
        metrics/ models/model_card.json models/production_eval.json 2>/dev/null || true
if git diff --cached --quiet; then
  echo "nothing changed - no ${kind} release"
  exit 0
fi
git -c user.name="${GIT_AUTHOR_NAME:-avdata-pipeline}" \
    -c user.email="${GIT_AUTHOR_EMAIL:-pipeline@avdata.local}" \
    commit -m "${kind} release ${stamp}" -m "$(dvc metrics show --md 2>/dev/null | head -40 || true)"
git tag "${kind}-${stamp}"
echo "tagged ${kind}-${stamp}"
if git remote get-url origin >/dev/null 2>&1 && [ "${AVDATA_GIT_PUSH:-0}" = "1" ]; then
  git push origin HEAD --tags
fi
