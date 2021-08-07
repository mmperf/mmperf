# Single CPU Core Matrix Multiplication Benchmarks

This repository aims to benchmark Matrix Multiply (SGEMM) hand-tuned libraries and code generation stacks on a single thread on one CPU core. The focus will be on machine learning workloads so FP32 or smaller and irregular sizes of matrices. The goal is to expose high performance atomic kernels that can then be used to build highly efficient higher level implemenations spanning multiple cores or distributed across systems. 

## Results

### Results on Intel XEON Skylake (iMAC PRO, AVX512)
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/skylake-avx512/2021-01-31_19-11-51-528540/matmul.png)

### Results on Xeon Cascade Lake (GCP C2 instance, AVX 512)
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/cascadelake/2021-01-31_15-47-19-968148/matmul.png)

### Results on Xeon Cascade Lake Codegen TVM, Halide, MLIR (GCP C2 instance, AVX 512)
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/skylake-avx512/2021-02-03_21-27-25-624537/matmul.png)

### Results on AMD Ryzen 5950x (ZenV3, compared to AMD's BLIS and OpenBLAS for RESNET50 sizes)
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/znver2/2021-01-29_16-16-24-502902/matmul.png)

### Results on Intel XEON E-2276M Coffee lake (Thinkpad P53, AVX2)
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/haswell/2021-02-03_14-06-35-488724/matmul.png)

### Results on Apple M1 (NEON - no AMX2)
Note: 8GB Mac Mini runs roughly 25% slower than the 16GB version on other tests.
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/apple-a13/2021-01-26_15-39-08/matmul.png)

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
cmake -GNinja \
    -DCMAKE_CXX_COMPILER=clang++-11 \
    -DCMAKE_C_COMPILER=clang-11 \
    -DMKL_DIR=/home/foo/lokal/intel/oneapi/mkl/latest/ \
    -DHAILE_DIR=/home/foo/lokal/halide \
    -DUSE_MLIR=ON -DUSE_MKL=ON -DUSE_RUY=ON -DUSE_HALIDE=ON -DUSE_OPENBLAS=ON \
    -B build .

cmake --build build
```

#### Compiling with a separately installed `llvm` distribution

Another example could be that you don't want to fetch and compile the source for `llvm-project` in the `external` folder, since you might already have your own compiled `llvm` distribution on your system. In that case, you can specify the path to your `llvm` distribution with `-DLLVM_DIR`:

```
cmake -GNinja \
    -DCMAKE_CXX_COMPILER=clang++-11 \
    -DCMAKE_C_COMPILER=clang-11 \
    -DLLVM_DIR=$HOME/opt/llvm \
    -DUSE_MLIR=ON \
    -B build .

cmake --build build
```

To plot the results, you will need to install matplotlib.

```
pip install matplotlib
```

If you want to compile the code with `llvm`, you might need these:

```bash
echo "deb http://apt.llvm.org/DISTRO_NAME/ llvm-toolchain-DISTRO_NAME main" >> /etc/apt/sources.list
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - 
apt-get update && apt-get upgrade -y
apt-get install -y clang-11 clang-tools-11 libc++1-11 libc++-11-dev \
    libc++abi1-11 libc++abi-11-dev libclang1-11 libclang-11-dev \
    libclang-common-11-dev libclang-cpp11 libclang-cpp11-dev liblld-11 \
    liblld-11-dev liblldb-11 liblldb-11-dev libllvm11 libomp-11-dev \
    libomp5-11 lld-11 lldb-11 llvm-11 llvm-11-dev llvm-11-runtime \
    llvm-11-tools libfuzzer-11-dev
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
 LLD_DIR=/usr/lib/llvm-11/lib/cmake/lld cmake . -GNinja -DCMAKE_BUILD_TYPE=Release -DTARGET_WEBASSEMBLY=OFF -DCMAKE_INSTALL_PREFIX=/home/<foo>/lokal/
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

