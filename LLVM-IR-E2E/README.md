To run examples you need to have:
- tf-opt
- mlir-hlo-opt
- tf-mlir-translate
- tf\_to\_kernel
- tf\_to\_gpu\_binary
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
bazel build //tensorflow/compiler/mlir:tf-mlir-translate && bazel build //tensorflow/compiler/mlir:tf
-opt && bazel build //tensorflow/compiler/mlir/tools/kernel_gen:kernel-gen-opt && bazel build //tenso
rflow/compiler/mlir/hlo:mlir-hlo-opt && bazel build //tensorflow/compiler/mlir/tools/kernel_gen:tf_to
_kernel && bazel build //tensorflow/compiler/mlir/tools/kernel_gen:tf_to_gpu_binary
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
**TF Absolute CPU example(![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) working)**
```sh
cd tf_abs
path/to/tf_to_kernel --input=tf_abs.mlir --output=abs_kernel.o
```

**TF Absolute GPU example(![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) broken)**
```sh
cd tf_abs
/path/to/tf_to_gpu_binary --input=tf_abs.mlir --output=abs_gpu_kernel.o
```

**MNIST CPU example(![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) broken)**
```sh
cd tf_mnist
path/to/tf_to_kernel --input=examples.minst_xla.mlir --output=mnist_kernel.o
```

**MNIST GPU example(![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) broken)**
```sh
cd tf_mnist
path/to/tf_to_gpu_binary --input=mnist_example/examples.mnist_xla.mlir --output=mnist_gpu_kernel.o
```
