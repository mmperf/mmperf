#!/usr/bin/env bash

set -e
set -o pipefail

cd matmul
cmake --build . -- run_all_tests
