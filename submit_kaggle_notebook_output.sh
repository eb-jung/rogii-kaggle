#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./submit_kaggle_notebook_output.sh <notebook-slug> <version> [message]

Example:
  ./submit_kaggle_notebook_output.sh fle3n-rogii-v5-train 1 "FLE3N LightGBM submission"

This wraps:
  kaggle competitions submit -c rogii-wellbore-geology-prediction \
    -f submission.csv -k e5jung/<notebook-slug> -v <version> -m "<message>"

Authentication:
  Export KAGGLE_API_TOKEN for KGAT access-token auth, or configure kaggle.json.
EOF
}

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Error: kaggle CLI is not installed or not on PATH." >&2
  echo "Install/configure the Kaggle CLI, then rerun this wrapper." >&2
  exit 127
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

NOTEBOOK_SLUG="$1"
VERSION="$2"
MESSAGE="${3:-FLE3N LightGBM submission}"

kaggle competitions submit \
  -c rogii-wellbore-geology-prediction \
  -f submission.csv \
  -k "e5jung/${NOTEBOOK_SLUG}" \
  -v "${VERSION}" \
  -m "${MESSAGE}"
