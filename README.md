# Single CPU Core Matrix Multiplication Benchmarks

This repository aims to benchmark Matrix Multiply (SGEMM) hand-tuned libraries and code generation stacks on a single thread on one CPU core. The focus will be on machine learning workloads so FP32 or smaller and irregular sizes of matrices. The goal is to expose high performance atomic kernels that can then be used to build highly efficient higher level implemenations spanning multiple cores or distributed across systems. 


### Sample results on Intel(R) Xeon(R) E-2276M Coffeelake (Thinkpad P53, AVX2)
![Results](https://github.com/mmperf/mmperf/blob/main/official_results/haswell/2021-01-24_15-25-42/matmul.png)

### Sample results on AMD Threadripper Pro 3990x (ZenV2, AVX2)
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/znver2/2021-01-24_15-58-48/matmul.png)

### Installation
First checkout the repo with submodules

```
git clone --recurse-submodules https://github.com/mmperf/mmperf.git
```

To build the code, run

```
cmake -GNinja -DCMAKE_CXX_COMPILER=clang++-11 -DCMAKE_C_COMPILER=clang-11 -DUSE_MLIR=ON -B build .
cmake --build build
```

Another example to build with all backends. Assumes you have MKL, OpenBLAS and Halide installed (see below to install) 

```
HALIDE_DIR=/home/foo/lokal/halide/ MKL_DIR=/opt/intel/oneapi/mkl/latest/ cmake -GNinja -DCMAKE_CXX_COMPILER=clang++-11 -DCMAKE_C_COMPILER=clang-11 -DMKL_DIR=/opt/intel/oneapi/mkl/latest/ -DUSE_MLIR=ON -DUSE_MKL=ON -DUSE_RUY=ON -DUSE_HALIDE=ON -DUSE_OPENBLAS=ON -B build .

cmake --build build
```

To plot the results, you will need to install matplotlib.

```
pip install matplotlib
```

### Running the code

We use AOT compilation to generate the binaries for matrix multiplication
and then run them to generate the benchmarking numbers. To run all the tests, do

```
cmake --build build/matmul --target run_all_tests
```

To plot the results against MKL (and generate a plot like above), run

```
python3 plot_results.py
```

To run a specific matrix size (say 24x64x512), run

```
./build/matmul/matmul_24x64x512
```

### Installing optional dependencies: Halide, OpenBLAS, MKL

#### Halide
```
 git clone https://github.com/halide/Halide.git --recurse-submodules
 cd Halide/
 sudo apt install libclang-11-dev clang-11 liblld-11-dev
 LLD_DIR=/usr/lib/llvm-11/lib/cmake/lld cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DTARGET_WEBASSEMBLY=OFF -DCMAKE_INSTALL_PREFIX=/home/<foo>/lokal/
 ninja
 ninja install
 export HALIDE_DIR=/home/<foo>/lokal/halide
```

#### OpenBLAS
```
sudo apt install libopenblas-dev
```

#### Intel MKL
Download and install from https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html

### Code structure

The linalg codegen pass is in matmul/matmul-compile/matmul-compile.cpp.

### Hardware information

This benchmark was run on an Intel Xeon CPU running at 3.1GHz. The machine has 256Kb L1 cache, 8Mb L2 cache and 24.8Mb L3 cache.
It supports AVX-512 instructions. The peak performance of the machine is 3.1 x 8 x 2 x 2 = 99.2 GFLOPS for double precision
and 198.4 GFLOPS for single precision.

### TODO:
Add Accelerate Framework
