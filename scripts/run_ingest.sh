#!/usr/bin/env bash
# Rebuilds the vector index from whatever is in data/materials/.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
python -m studybot.ingest
