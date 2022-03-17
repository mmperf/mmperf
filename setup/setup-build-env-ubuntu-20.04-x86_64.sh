#!/usr/bin/env bash

# Installs system-wide build dependencies and
# sets up a python venv in the current working directory. 

set -e
set -o pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

sudo apt-get install \
  clang-12 \
  lld-12 \
  ninja-build \
  python3.9 \
  wget

python3.9 -m venv .
pip install -r "$SCRIPT_DIR/../requirements.txt"

wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4-linux-x86_64.sh \
  -O /tmp/cmake-3.18.4-linux-x86_64.sh
bash /tmp/cmake-3.18.4-linux-x86_64.sh --prefix=. --exclude-subdir --skip-license
rm /tmp/cmake-3.18.4-linux-x86_64.sh
