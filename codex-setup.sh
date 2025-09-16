#!/usr/bin/env bash
set -euxo pipefail

# System deps (Ubuntu 24.04)
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-dev build-essential pkg-config curl ca-certificates \
  # OpenImageIO (Python bindings + tools)
  python3-openimageio libopenimageio-dev openimageio-tools \
  # Assimp FBX loader + Python bindings
  libassimp-dev assimp-utils python3-pyassimp

# Create a venv that can see APT-installed Python packages
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# Project Python deps
python -m pip install -r requirements.txt

# Install the package in editable mode if present (safe on first run too)
if [ -f pyproject.toml ]; then
  python -m pip install -e .
fi

# Ensure the venv auto-activates for Codex agent shells
if ! grep -q "source $(pwd)/.venv/bin/activate" "${HOME}/.bashrc"; then
  echo "source $(pwd)/.venv/bin/activate" >> "${HOME}/.bashrc"
fi

# Quick sanity check
python - <<'PY'
import sys
print("Python:", sys.version.split()[0])
try:
    import OpenImageIO as oiio
    print("OpenImageIO:", getattr(oiio, "__version__", "import-ok"))
except Exception as e:
    print("OpenImageIO: import-failed", e)
try:
    import pyassimp
    print("pyassimp: import-ok")
except Exception as e:
    print("pyassimp: import-failed", e)
PY
