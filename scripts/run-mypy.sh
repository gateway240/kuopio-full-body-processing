#!/usr/bin/env bash

# A script for running mypy,
# with all its dependencies installed.

set -o errexit

echo "$(pwd)"
source ./.venv/bin/activate
mypy
