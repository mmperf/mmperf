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
  python3.9-venv \
  wget

python3.9 -m venv .
source bin/activate
pip install -r "$SCRIPT_DIR/../requirements.txt"

wget https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2-linux-x86_64.sh \
  -O /tmp/cmake-3.22.2-linux-x86_64.sh
bash /tmp/cmake-3.22.2-linux-x86_64.sh --prefix=. --exclude-subdir --skip-license
rm /tmp/cmake-3.22.2-linux-x86_64.sh
