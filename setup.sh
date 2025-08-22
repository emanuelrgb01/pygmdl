#!/usr/bin/env bash
set -e

PYTHON_VERSION=3.12
VENV_DIR=".venv"

python$PYTHON_VERSION -m venv $VENV_DIR

source $VENV_DIR/bin/activate

pip install uv pip-tools

pip-compile requirements.in --output-file requirements.txt --upgrade

uv pip install -r requirements.txt

uv pip install -e .

