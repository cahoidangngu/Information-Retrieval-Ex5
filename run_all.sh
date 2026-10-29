#!/usr/bin/env bash
# Reproduces everything from raw data: download -> tests -> Stage1/2 -> models -> analysis -> ablation -> cross-check
# -> EXTENSIONS (hinge SVM, feature sets v1/v1_reduced/v2, 5-fold CV, stemmed Stage 1) -> report.
# Usage:  PYTHON=/path/to/python bash run_all.sh [--smoke]     (default PYTHON=python; needs the packages in environment.yml)
# Runtime: roughly 3 min for the original pipeline + ~10 min hinge + ~3 min feature sets + ~15 min CV + <1 min stemmed Stage 1 (see results/total_runtime_seconds.txt).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
SMOKE="${1:-}"
START=$(date +%s)
mkdir -p results data/processed
"$PY" --version
# -I = isolated mode (no cwd / user-site on sys.path); entry scripts add src/ explicitly.
"$PY" -I src/run_features.py $SMOKE
"$PY" -m pytest -q tests                      # needs results from the previous step for the saved-scaler test
"$PY" -I src/run_models.py $SMOKE
"$PY" -I src/analysis.py $SMOKE
"$PY" -I src/ablation.py $SMOKE
"$PY" -I src/crosscheck.py $SMOKE || echo "crosscheck skipped/failed (ranx missing?)"
# ---- extensions (separate scripts; write results/ext_*.csv, never modify the files above)
"$PY" -I src/run_hinge.py $SMOKE
"$PY" -I src/run_featuresets.py $SMOKE
"$PY" -I src/run_cv.py $SMOKE
"$PY" -I src/run_stemmed_stage1.py $SMOKE
if [ -z "$SMOKE" ]; then
  "$PY" -m pytest -q tests                    # again, now that all results exist
  echo $(( $(date +%s) - START )) > results/total_runtime_seconds.txt   # first estimate, refreshed below
  "$PY" -I src/make_report.py
  echo $(( $(date +%s) - START )) > results/total_runtime_seconds.txt
  "$PY" -I src/make_report.py                 # regenerate so the report quotes the final runtime
fi
echo "done in $(( $(date +%s) - START )) s"
