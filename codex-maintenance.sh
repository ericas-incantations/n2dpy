#!/usr/bin/env bash
set -euxo pipefail
source .venv/bin/activate
python -m pip install -U -r requirements.txt
if [ -f pyproject.toml ]; then
  python -m pip install -e .
fi
