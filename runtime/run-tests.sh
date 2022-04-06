#!/bin/bash
set -e

echo "Running Python tests"
conda run -n py python tests/test_installs.py
