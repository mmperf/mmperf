# LLVM-IR-E2E
End to end examples on generating LLVM IR code from high level MLIR  

### Prerequisistes:

To run examples you need to have: 
- tf-opt
- mlir-hlo-opt
- tf-mlir-translate
- kernel-gen-opt
- mlir-opt
- mlir-tranlsate

using TF: da034876d3d1849f6328c9c05c918b506e62dd63\
using LLVM: 0f6facca9701f6df87d13e55d70bd7111a0472aa

**Intalling TensorFlow Opt and Tools**:
```sh
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
./configure #say N to every [y/N] execpt CUDA(that's up to you :))
bazel build //tensorflow/compiler/mlir:tf-mlir-translate && bazel build //tensorflow/compiler/mlir:tf-opt && bazel build //tensorflow/compiler/mlir/tools/kernel_gen:kernel-gen-opt && bazel build //tensorflow/compiler/mlir/hlo:mlir-hlo-opt
```
**Intalling MLIR Tools**:
```sh
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
#  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
cmake --build . --target check-mlir
```

### How to run:
**TF Absolute example**
```sh
cd tf_abs
./abs_gen.sh tf_abs.mlir output.bc path/to/mlir-hlo-opt path/to/tf-opt path/to/kernel-gen-opt path/to/mlir-translate
```

**MNIST example**
```sh
cd tf_mnist
./mnist_gen.sh examples.mnist_xla.mlir output.bc path/to/mlir-hlo-opt path/to/tf-opt path/to/kernel-gen-opt path/to/mlir-translate
```
