#!/usr/bin/env bash
# Runs once, after the container is created.
set -euo pipefail

# Named volumes are created empty and owned by root; hand ~/.claude to the
# container user so the CLI can write its config and credentials.
if [ -d "$HOME/.claude" ] && [ ! -w "$HOME/.claude" ]; then
  sudo chown -R "$(id -u):$(id -g)" "$HOME/.claude"
fi

# `python` is /opt/venv/bin/python (3.14, with the cu132 PyTorch build).
python -m pip install --upgrade pip

if [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
fi
