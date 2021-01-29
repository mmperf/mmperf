# Single CPU Core Matrix Multiplication Benchmarks

This repository aims to benchmark Matrix Multiply (SGEMM) hand-tuned libraries and code generation stacks on a single thread on one CPU core. The focus will be on machine learning workloads so FP32 or smaller and irregular sizes of matrices. The goal is to expose high performance atomic kernels that can then be used to build highly efficient higher level implemenations spanning multiple cores or distributed across systems. 


### Results on AMD Ryzen 5950x (ZenV3, compared to AMD's BLIS and OpenBLAS for RESNET50 sizes)
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/znver2/2021-01-29_06-25-31-351049/matmul.png)

### Results on Intel XEON Skylake (GCP C2 instance, AVX512)
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/skylake-avx512/2021-01-26_01-12-27/matmul.png)

### Results on AMD Threadripper 3990x (ZenV2, AVX2)
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/znver2/2021-01-25_13-24-25/matmul.png)

### Results on Intel XEON E-2276M Coffee lake (Thinkpad P53, AVX2)
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/haswell/2021-01-26_16-42-20/matmul.png)

### Results on Apple M1 (NEON - no AMX2)
Note: 8GB Mac Mini runs roughly 25% slower than the 16GB version on other tests.
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/apple-a13/2021-01-26_15-39-08/matmul.png)

### Results on Apple M1 (RUY/MLIR using NEON - Accelerate with AMX2)
Note 0: 8GB Mac Mini runs roughly 25% slower than the 16GB version on other tests.
Note 1: Set veclib_maximum_threads=1 but there is no way to verify it is honored by Accelerate. 
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/apple-a13/2021-01-26_18-33-13/matmul.png)

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

The plot will be saved in matmul.png 

To run a specific matrix size (say 24x64x512), run

```
./build/matmul/matmul_<LIBRARY>_24x64x512
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

### BLIS
```
git clone https://github.com/flame/blis
cd blis
./configure --prefix=/home/foo/lokal/ --enable-cblas -c amd64
make -j 16
make install
```

#### Intel MKL
Download and install from https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html

### Code structure

The linalg codegen pass is in matmul/matmul-compile/matmul-compile.cpp.

### Theoretical Max FLOPS 

This benchmark was run on an Intel Xeon CPU running at 3.1GHz. The machine has 256Kb L1 cache, 8Mb L2 cache and 24.8Mb L3 cache.
It supports AVX-512 instructions. The peak performance of the machine is 3.1 x 8 x 2 x 2 = 99.2 GFLOPS for double precision
and 198.4 GFLOPS for single precision.

