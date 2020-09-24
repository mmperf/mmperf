#!/bin/bash
# Usage (for in-tree build/ directory):
#   ./install_deps.sh
# Usage (for aribtrary build/ directory):
#   BUILD_DIR=/build ./install_deps.sh
set -e
td="$(realpath $(dirname $0))"
build_dir="$(realpath "${BUILD_DIR:-$td/build}")"

# Find LLVM source (assumes it is adjacent to this directory).
LLVM_SRC_DIR="$(realpath "${LLVM_SRC_DIR:-$td/external/llvm-project}")"
TF_SRC_DIR="$(realpath "${TF_SRC_DIR:-$td/external/tensorflow}")"

if ! [ -f "$LLVM_SRC_DIR/llvm/CMakeLists.txt" ]; then
  echo "Expected LLVM_SRC_DIR variable to be set correctly (got '$LLVM_SRC_DIR')"
  exit 1
fi
echo "Using LLVM source dir: $LLVM_SRC_DIR"
echo "Build directory: $build_dir"
# Setup directories.
build_mlir="$build_dir/build-mlir"
install_mlir="$build_dir/install"
echo "Building MLIR in $build_mlir"
echo "Install MLIR to $install_mlir"
mkdir -p "$build_mlir"
mkdir -p "$install_mlir"

echo "Beginning build (commands will echo)"
set -x

echo "Beginning MLIR build (commands will echo)"
# TODO: Make it possible to build without an RTTI compiled LLVM. There are
# a handful of vague linkage issues that need to be fixed upstream.
cmake -GNinja \
  "-H$LLVM_SRC_DIR/llvm" \
  "-B$build_mlir" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AArch64;ARM" \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  "-DCMAKE_INSTALL_PREFIX=$install_mlir" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DLLVM_ENABLE_RTTI=On

cmake --build "$build_mlir" --target install
echo "Done building MLIR"


echo "Beginning Tensorflow build (commands will echo)"
echo "Build directory: $build_dir"

echo "Configure your TF build - typically you just want CUDA enabled $build_mlir"
cd $TF_SRC_DIR
if [ ! -f ".tf_configure.bazelrc" ]; then
    echo ".configure does not exist. Running ./configure"
    ./configure
fi
echo "TF configured... invoking bazel. If you dont have bazel .. download and install bazelisk"
bazel build //tensorflow/compiler/mlir:tf-mlir-translate //tensorflow/compiler/mlir:tf-opt //tensorflow/compiler/mlir/tools/kernel_gen:kernel-gen-opt //tensorflow/compiler/mlir/hlo:mlir-hlo-opt //tensorflow/compiler/mlir/tools/kernel_gen:tf_to_kernel //tensorflow/compiler/mlir/tools/kernel_gen:tf_to_gpu_binary

ln -s ${TF_SRC_DIR}/bazel-bin/tensorflow/compiler/mlir/tf-mlir-translate $install_mlir/bin/
ln -s ${TF_SRC_DIR}/bazel-bin/tensorflow/compiler/mlir/tf-opt $install_mlir/bin/
ln -s ${TF_SRC_DIR}/bazel-bin/tensorflow/compiler/mlir/hlo/mlir-hlo-opt $install_mlir/bin/
ln -s ${TF_SRC_DIR}/bazel-bin/tensorflow/compiler/mlir/tools/kernel_gen/kernel-gen-opt $install_mlir/bin/
ln -s ${TF_SRC_DIR}/bazel-bin/tensorflow/compiler/mlir/tools/kernel_gen/tf_to_kernel $install_mlir/bin/
ln -s ${TF_SRC_DIR}/bazel-bin/tensorflow/compiler/mlir/tools/kernel_gen/tf_to_gpu_binary $install_mlir/bin/
echo "All done and installed in $install_mlir"
