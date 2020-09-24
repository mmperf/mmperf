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

# Intalling LLVM/MLIR and TF Dependencies:
```sh
git clone https://github.com/NodLabs/mlir-examples.git
mkdir mlir-examples/
./install_deps.sh
```
Export the pre-built binaries into your path (you can also directly call the binaries)

```
export PATH=`pwd`/build/install/bin:$PATH
```


# Generating MHLO->LLVM-IR->Target examples

Now cd into one of the examples and give it a try. 

```
./build/install/bin/tf_to_kernel --input=LLVM-IR-E2E/tf_abs/tf_abs.mlir --output=abs_kernel.o

file abs_kernel.o
```

# Generating MHLO->C++->Target examples

```
mkdir build/emitc
cd build/emitc
cmake -G Ninja -DCMAKE_C_COMPILER=clang-10 -DCMAKE_CXX_COMPILER=clang++-10 .. -DCMAKE_PREFIX_PATH=../../build/install/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=`pwd`/../build-mlir/bin/llvm-lit ../../external/mlir-emitc/
cmake --build . --target check-emitc
```
Currently you will notice three tests failing

```
-- Testing: 13 tests, 13 workers --
UNRESOLVED: EMITC :: testCorrectGroundTruthWithHMC_canon_inline.mlir (1 of 13)
UNRESOLVED: EMITC :: testCorrectGroundTruthWithHMC_canon.mlir (2 of 13)
PASS: EMITC :: Target/cpp-calls-for.mlir (3 of 13)
UNRESOLVED: EMITC :: testCorrectGroundTruthWithHMC.mlir (4 of 13)
PASS: EMITC :: Dialect/EmitC/ops.mlir (5 of 13)
PASS: EMITC :: Target/cpp-calls.mlir (6 of 13)
PASS: EMITC :: Target/cpp-calls-if.mlir (7 of 13)
PASS: EMITC :: Conversion/mhlo-to-std.mlir (8 of 13)
PASS: EMITC :: Conversion/std-to-emitc.mlir (9 of 13)
PASS: EMITC :: Conversion/mhlo-to-emitc.mlir (10 of 13)
PASS: EMITC :: Conversion/scf-to-emitc.mlir (11 of 13)
PASS: EMITC :: Dialect/EmitC/ifop.mlir (12 of 13)
PASS: EMITC :: Dialect/EmitC/forop.mlir (13 of 13)
********************
Unresolved Tests (3):
  EMITC :: testCorrectGroundTruthWithHMC.mlir
  EMITC :: testCorrectGroundTruthWithHMC_canon.mlir
  EMITC :: testCorrectGroundTruthWithHMC_canon_inline.mlir


Testing Time: 0.12s
  Passed    : 10
  Unresolved:  3
FAILED: test/CMakeFiles/check-emitc
cd /home/anush/github/mlir-examples/build/emitc/test && /home/anush/github/mlir-examples/build/emitc/../build-mlir/bin/llvm-lit /home/anush/github/mlir-examples/build/emitc/test
ninja: build stopped: subcommand failed.
```




# Other references
https://llvm.discourse.group/t/print-in-mlir/1701/13
