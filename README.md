# Collection of easy to use MLIR end to end examples. 
A collection of MLIR End to End examples. We plan to add working examples of:

1. TF2 Saved Model --> MHLO --> C/C++
  *  --> C++ --> target of your choice. With EmitC: (https://github.com/iml130/mlir-emitc/tree/cgo) 
  *  --> C++ --> target of your device. With TCIE: (https://gist.github.com/silvasean/8eb20f3649855b64f525c7141033f053)
2. TF2 Saved Model --> MHLO --> LLVM_IR
  * --> RefE2E (?)
  * --> LLVM-IR-e2e in this repo
3. Torchscript --> ATen --> [C++ / LLVM_IR]
  * Use mlir-npcomp to generate ATen Dialect. 
  * Lower to C++ / LLVM_IR (?)

**Intalling LLVM/MLIR and TF Dependencies**:
```sh
git clone https://github.com/NodLabs/mlir-examples.git
mkdir mlir-examples/
./install_deps.sh
```
Export the pre-built binaries into your path (you can also directly call the binaries)

```
export PATH=`pwd`/build/install/bin:$PATH
```

Now cd into one of the examples and give it a try. 

```
./build/install/bin/tf_to_kernel --input=LLVM-IR-E2E/tf_abs/tf_abs.mlir --output=abs_kernel.o

file abs_kernel.o
```

# Other references
https://llvm.discourse.group/t/print-in-mlir/1701/13
