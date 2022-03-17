#!/usr/bin/env bash

set -e
set -o pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

VALID_ARGS=$(getopt --long build-type: -- "$@")

eval set -- "$VALID_ARGS"
while [ $# -gt 0 ]
do
  case "$1" in
    --build-type)
      BUILD_TYPE="$2"
      shift
      ;;
    (--) shift; break;;
    (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1;;
    (*) break;;
  esac
  shift
done

cmake \
  -GNinja \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_C_COMPILER=clang-12 \
  -DCMAKE_CXX_COMPILER=clang++-12 \
  -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld-12" \
  -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld-12" \
  -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld-12" \
  -DUSE_MLIR=ON \
  -DUSE_IREE=ON \
  -DIREE_DYLIB=ON \ 
  -S "$SCRIPT_DIR/.."
