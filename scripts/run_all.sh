#!/usr/bin/env bash
# End-to-end: tests -> S1–S6 pipeline -> M0–M5 models -> C1–C6 evaluation.
#
#   bash scripts/run_all.sh                    # download via kagglehub (public dataset)
#   bash scripts/run_all.sh --source PATH      # use a manually downloaded zip / train.csv
#   bash scripts/run_all.sh --skip-acquire     # reuse data/raw/delivery_raw.csv
#   bash scripts/run_all.sh --refreeze         # features changed: rebuild split (then retrain)
#
# Then:  streamlit run app/streamlit_app.py
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
export PYTHONWARNINGS="ignore::RuntimeWarning"   # spurious numpy+Accelerate matmul warnings

echo "== tests (synthetic fixture only) =="
"$PY" -m pytest

echo "== S1–S6 pipeline =="
"$PY" -m src.pipeline.run "$@"

echo "== M0–M5 models =="
"$PY" -m src.models.train

echo "== C1–C6 evaluation =="
"$PY" -m src.evaluation.evaluate

echo
echo "Done. Launch the dashboard with:  streamlit run app/streamlit_app.py"
