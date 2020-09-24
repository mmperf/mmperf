**Install Deps**
See here to install deps via ./install_deps.sh
https://github.com/NodLabs/mlir-examples

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
