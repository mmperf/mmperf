# Single CPU Core Matrix Multiplication Benchmarks

This repository aims to benchmark Matrix Multiply (SGEMM) hand-tuned libraries and code generation stacks on a single thread on one CPU core. The focus will be on machine learning workloads so FP32 or smaller and irregular sizes of matrices. The goal is to expose high performance atomic kernels that can then be used to build highly efficient higher level implemenations spanning multiple cores or distributed across systems.

## Results

### Results on Nvidia A100 (cublas vs SHARK)
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/cuda_nodai7.png)

### Results on Intel Alderlake 12900k (AVX2)
![Results](https://github.com/mmperf/mmperf/raw/main/official_results/alderlake/2022-01-27_23-40-34-673036/matmul.png)

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

## Installation
Clone the repo along with submodules.

```
git clone --recurse-submodules https://github.com/mmperf/mmperf.git
```

Create a virtual environment and install requirements. Python 3.8 is required.
```
cd mmperf
python3 -m venv ./mmperf_env
source mmperf_env/bin/activate
pip install -r requirements.txt
pip install -r ./external/llvm-project/mlir/python/requirements.txt

```

## Building the codes
### Building specified backends on CPU
Build the project specifying the backend(s) to run matmul. Below is a command to build mmperf with MLIR backend.

```
cmake -GNinja \
    -DCMAKE_CXX_COMPILER=clang++-11 \
    -DCMAKE_C_COMPILER=clang-11 \
    -DUSE_MLIR=ON \
    -B build .

cmake --build build
```

Another example to build with all available backends. Assumes you have MKL, OpenBLAS, and Halide installed (see below for installation details)

```
HALIDE_DIR=/home/foo/lokal/halide/ MKL_DIR=/opt/intel/oneapi/mkl/latest/ cmake -GNinja \
    -DCMAKE_CXX_COMPILER=clang++-11 \
    -DCMAKE_C_COMPILER=clang-11 \
    -DMKL_DIR=/opt/intel/oneapi/mkl/latest/ \
    -DUSE_MLIR=ON \
    -DUSE_MKL=ON \
    -DUSE_RUY=ON \
    -DUSE_HALIDE=ON \
    -DUSE_OPENBLAS=ON \
    -DUSE_IREE=ON \
    -DIREE_DYLIB=ON \
    -B build .

cmake --build build
```

### Building specified backends on GPU
With any GPU backends, NVIDIA CUDA-11.4 should be pre-installed on your system. To install NVIDIA CUDA-11.4 toolkit, please refer to this [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Make sure environment variables `$PATH` and `$LD_LIBRARY_PATH` are correctly configured. CUDA Compiler should be set as `nvcc` in the command line. For example, to compile the MLIR-CUDA backend run:

```bash
cmake -GNinja \
    -DCMAKE_CXX_COMPILER=clang++-11 \
    -DCMAKE_C_COMPILER=clang-11 \
    -DCMAKE_CUDA_COMPILER=nvcc \
    -DUSE_MLIR_CUDA=ON \
    -B build .

cmake --build build
```

Another example to compare mmperf results among cuBLAS, IREE-CUDA and TVM-CUDA (TVM Auto-scheduler, a.k.a. Ansor) with this command line:

```bash
cmake -GNinja \
    -DCMAKE_CXX_COMPILER=clang++-11 \
    -DCMAKE_C_COMPILER=clang-11 \
    -DCMAKE_CUDA_COMPILER=nvcc \
    -DUSE_IREE=ON \
    -DIREE_CUDA=ON \
    -DUSE_CUBLAS=ON \
    -DUSE_TVM_CUDA=ON \
    -DTVM_ENABLE_CUDA=ON \
    -DUSE_TVM_TUNED=ON \
    -DTVM_LIB_DIR=/path/to/tvm-tuner
    -DSIZE_FILE=benchmark_sizes/bert_large_matmul.txt 
    -B build .
```
Note: `-DTVM_LIB_DIR` should be set as the absolute path of where TVM binaries located. For how to run TVM auto-scheduler, please refer to this [README](tvm_tuner/README.md).

### Building with a standalone `llvm` (optional)

The building of submodule `external/llvm-project` can be space and time consuming. If you already have your own standalone `llvm` and don't want to fetch and compile this submodule, you scan specify the `llvm` on your system with `PREBUILT_LLVM_PATH` compilation flag:

```bash
cmake -GNinja \
    -DCMAKE_CXX_COMPILER=clang++-11 \
    -DCMAKE_C_COMPILER=clang-11 \
    -DPREBUILT_LLVM_PATH=$HOME/opt/llvm \
    -DUSE_MLIR=ON \
    -B build .

cmake --build build
```

To compile `llvm` from scratch, you might want all of these as well:

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

## Running and generating results

We use AOT compilation to generate binaries for matrix multiplication for specified backends
and run them to generate the benchmarking numbers. To generate performance numbers and get a comparison plot run:

```
python3 mmperf.py ./build/matmul/ results
```

`results` folder will be created in the mmperf top-level directory which will contain GLOPS for every matmul size and every backend. A plot comparing performances of backends will also be generated in `matmul.png`.

Each generated binary can also be executed individually. To run a specific matrix size (say 24x64x512) for a backend run:

```
./build/matmul/matmul_<LIBRARY>_24x64x512
```

### Program configuration

Matrix sizes: `benchmark_sizes` folder has text files containing the matrix sizes that mmperf runs on. You can change the matrix size input file by editing `SIZE_FILE` option in `cmake/common.cmake`. Default is `benchmark_all_sizes.txt`.

Precision: The default precision for all backends is FP32. FP16 benchmark has been added to cuBLAS, IREE, and triton backends. To enable FP16, use flag `-DUSE_FP16=ON`


### Run iree-llvm-sandbox in mmperf
To build mlir with iree-llvm-sandbox, enable the flag `-DUSE_IREE_LLVM_SANDBOX=ON`.

To run iree-llvm-sandbox, and get the best performance among different experts, run:
```
cmake --build build/matmul --target sandbox_search
```
Note: This command will perform search on all expert configurations, and output the best configs to `mlir_sandbox_configs` for each matmul size.

To compare results between original mlir-sandbox and nodai-sandbox, run:
```
python mmperf.py ./build/matmul results -sandbox -sandbox_configs=/path/to/mlir_sandbox_configs -nodai_configs=/path/to/nodai_sandbox_configs
```

Note: If you just want to plot results for mlir_sandbox, enable `-sandbox_configs` (to load configs from previous run) or `-benchmark_path` (to rerun the search). Optional flag: `-num_iters` to change the number of iterations for matmul test.

### Run triton with FP16
```
python mmperf.py ./build/matmul results -triton -dtype f16 -benchmark_path=./benchmark_sizes/bert_large_matmul.txt
```

## Installing optional dependencies: Halide, OpenBLAS, MKL

### Halide
```
git clone https://github.com/halide/Halide.git --recurse-submodules
cd Halide/
sudo apt install libclang-11-dev clang-11 liblld-11-dev
LLD_DIR=/usr/lib/llvm-11/lib/cmake/lld cmake . -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTARGET_WEBASSEMBLY=OFF \
    -DCMAKE_INSTALL_PREFIX=/home/<foo>/lokal/
ninja
ninja install
export HALIDE_DIR=/home/<foo>/lokal/halide
```

### OpenBLAS
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

### Intel MKL
Download and install from https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html

## Theoretical Max FLOPS
This benchmark was run on an Intel Xeon CPU running at 3.1GHz. The machine has 256Kb L1 cache, 8Mb L2 cache and 24.8Mb L3 cache. It supports AVX-512 instructions. The peak performance of the machine is 3.1 x 8 x 2 x 2 = 99.2 GFLOPS for double precision and 198.4 GFLOPS for single precision.

