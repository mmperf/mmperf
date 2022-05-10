#ifdef MKL
#include "mkl.h"
#elif defined(BLASFEO)
#include <blasfeo_s_blas_api.h>
#elif defined(OPENBLAS) || defined(BLIS) || defined (ACCELERATE)
#include <cblas.h>
#elif defined(HALIDE)
#include "halide_blas.h"
#elif defined(RUY)
#include "ruy/ruy.h"
#elif defined(TVM)
#include <unordered_map>
#include "tvm/te/schedule_pass.h"
#include "tvm/te/schedule.h"
#include "tvm/te/operation.h"
#include "tvm/driver/driver_api.h"
#include "tvm/runtime/registry.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/module.h"
#include "tvm/target/codegen.h"
#elif defined(CUBLAS)
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <sstream>
#elif defined(MLIR_CUDA)
#include "cuda_runtime.h"
#include <sstream>
#elif defined(MLIR)
#endif
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <unistd.h>
#include <benchmark/benchmark.h>

#define STRING(s) #s
#define TO_STRING(x) STRING(x)

#ifdef COLUMN_MAJOR
#if defined(MKL) || defined(BLASFEO) || defined(OPENBLAS) || defined(BLIS) || defined(ACCELERATE)
#define MATRIX_FORMAT CblasColMajor
#elif defined(HALIDE)
#define MATRIX_FORMAT HblasColMajor
#elif defined(RUY)
#define MATRIX_FORMAT ruy::Order::kColMajor
#endif
#else
#if defined(MKL) || defined(BLASFEO) || defined(OPENBLAS) || defined(BLIS) || defined(ACCELERATE)
#define MATRIX_FORMAT CblasRowMajor
#elif defined(HALIDE)
#define MATRIX_FORMAT HblasRowMajor
#elif defined(RUY)
#define MATRIX_FORMAT ruy::Order::kRowMajor
#endif
#endif

#ifdef USE_FP16
  using dtype = half;
#else
  using dtype = float;
#endif

#if defined(MLIR) || defined(MLIR_CUDA)
extern "C" {
struct memref_t {
  float *aligned;
  float *allocated;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};
void matmul(float *aligned_a, float *allocated_a, int64_t offset_a,
            int64_t size_a0, int64_t size_a1, int64_t strides_a0, int64_t strides_a1,
            float *aligned_b, float *allocated_b, int64_t offset_b,
            int64_t size_b0, int64_t size_b1, int64_t strides_b0, int64_t strides_b1,
            float *aligned_c, float *allocated_c, int64_t offset_c,
            int64_t size_c0, int64_t size_c1, int64_t strides_c0, int64_t strides_c1);
}
#endif

#ifdef CUBLAS
#define CHECK_CUBLAS(status) do {				\
  std::stringstream error;					\
  if (status != CUBLAS_STATUS_SUCCESS) {			\
    printf("Error %d at %s:%d\n", status, __FILE__, __LINE__); 	\
    exit(1);							\
  }								\
} while(0)
#endif

#if defined(CUBLAS) || defined(MLIR_CUDA)
#define CHECK_CUDA(status) do {				        \
  std::stringstream error;					\
  if (status != cudaSuccess) {					\
    printf("Error %d at %s:%d\n", status, __FILE__, __LINE__); 	\
    exit(1);							\
  }								\
} while(0)
#endif

#if defined(TVM)|| defined(TVM_CUDA)
tvm::runtime::Module create_module() {
  // Define algorithm
  tvm::te::Tensor A = tvm::te::placeholder({MDIM, KDIM}, tvm::DataType::Float(32), "A");
  tvm::te::Tensor B = tvm::te::placeholder({KDIM, NDIM}, tvm::DataType::Float(32), "B");
  auto k = tvm::te::reduce_axis(tvm::Range{0, KDIM}, "k");
  // TODO: Add column major support to TVM
  int bn = 32;
  auto packedB = tvm::te::compute({NDIM / bn, KDIM, bn},
    [&](tvm::te::Var i, tvm::te::Var j, tvm::te::Var k) {
       return B[j][i * bn + k];
    }, "packedB");
  auto C = tvm::te::compute({MDIM, NDIM},
    [&](tvm::te::Var i, tvm::te::Var j) {
      return tvm::sum(A[i][k] * packedB[tvm::floordiv(j, bn)][k][tvm::indexmod(j, bn)], {k});
    }, "C");

  // Define schedule
  // This schedule is based on: https://tvm.apache.org/docs/tutorials/optimize/opt_gemm.html
  auto s = tvm::te::create_schedule({C->op});
  auto CC = s.cache_write(C, "global");

  tvm::te::IterVar x_outer, y_outer, x_inner, y_inner;
  auto cAxis = C->op.as<tvm::te::ComputeOpNode>()->axis;
  s[C].tile(cAxis[0], cAxis[1], bn, bn,
            &x_outer, &y_outer, &x_inner, &y_inner);
  s[CC].compute_at(s[C], y_outer);
  auto newAxis = s[CC]->op.as<tvm::te::ComputeOpNode>()->axis;
  auto xc = newAxis[0];
  auto yc = newAxis[1];
  auto kk = s[CC]->op.as<tvm::te::ComputeOpNode>()->reduce_axis;
  tvm::te::IterVar k_outer, k_inner;
  s[CC].split(kk[0], 4, &k_outer, &k_inner);
  s[CC].reorder({k_outer, xc, k_inner, yc});
  s[CC].unroll(k_inner);
  s[CC].vectorize(yc);

  s[C].parallel(x_outer);

  auto bAxis = s[packedB]->op.as<tvm::te::ComputeOpNode>()->axis;
  s[packedB].vectorize(bAxis[2]);
  s[packedB].parallel(bAxis[0]);

  auto args = tvm::Array<tvm::te::Tensor>({A, B, C});
  std::unordered_map<tvm::te::Tensor, tvm::te::Buffer> binds;

#ifdef TVM_ENABLE_CUDA
  auto target = tvm::Target("cuda");
#else
  auto target = tvm::Target("llvm");
#endif

  auto lowered = tvm::LowerSchedule(s, args, "matmul", binds);
  auto module = tvm::build(lowered, target, tvm::Target());
  return module;
}
#endif

void init_matrix(dtype *a, int nrows, int ncols) {
  for (int j = 0; j < ncols; j++) {
    for (int i = 0; i < nrows; i++) {
      #ifdef USE_FP16
        a[i + j * nrows] = __float2half(((float) rand() / (float) RAND_MAX));
      #else
        a[i + j * nrows] = ((float) rand() / (float) RAND_MAX);
      #endif
    }
  }
}

void init_matrix_batch(dtype *a, int batch, int nrows, int ncols) {
  for (int b = 0; b < batch; b++){
    for (int j = 0; j < ncols; j++) {
      for (int i = 0; i < nrows; i++) {
        #ifdef USE_FP16
          a[i + j * nrows + b * ncols * nrows] = __float2half((float) rand() / (float) RAND_MAX);
        #else
          a[i + j * nrows + b * ncols * nrows] = ((float) rand() / (float) RAND_MAX);
        #endif
      }
    }
  }
}

void naive_matmul(const dtype *a, const dtype *b, float *c, size_t m, size_t k, size_t n) {
  // correctness check
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
#ifdef COLUMN_MAJOR
      size_t ci = i + j*m;
#else
      size_t ci = i*n + j;
#endif
      c[ci] = 0.0f;
      for (size_t p = 0; p < k; p++) {
#ifdef COLUMN_MAJOR
        c[ci] += a[i + p*m] * b[p + j*k];
#else
        c[ci] += __half2float(a[i*k + p]) * __half2float(b[p*n + j]);
#endif
      }
    }
  }
}

static void BenchmarkFunction(benchmark::State& state) {
dtype *A, *B, *C;
const dtype *A_Array[BDIM];
const dtype *B_Array[BDIM];
dtype *C_Array[BDIM];

if (BDIM == 0){
#ifdef MLIR_CUDA
  CHECK_CUDA(cudaMallocHost((void **) &A, MDIM * KDIM * sizeof(float)));
  CHECK_CUDA(cudaMallocHost((void **) &B, KDIM * NDIM * sizeof(float)));
  CHECK_CUDA(cudaMallocHost((void **) &C, MDIM * NDIM * sizeof(float)));
#else
  A = (dtype *) malloc(MDIM * KDIM * sizeof(dtype));
  B = (dtype *) malloc(KDIM * NDIM * sizeof(dtype));
  C = (dtype *) malloc(MDIM * NDIM * sizeof(dtype));
#endif

  init_matrix(A, MDIM, KDIM);
  init_matrix(B, KDIM, NDIM);
  init_matrix(C, MDIM, NDIM);
}
else{
  A = (dtype *) malloc(BDIM * MDIM * KDIM * sizeof(dtype));
  B = (dtype *) malloc(BDIM * KDIM * NDIM * sizeof(dtype));
  C = (dtype *) malloc(BDIM * MDIM * NDIM * sizeof(dtype));
  init_matrix_batch(A, BDIM, MDIM, KDIM);
  init_matrix_batch(B, BDIM, KDIM, NDIM);
  init_matrix_batch(C, BDIM, MDIM, NDIM);
  for (int i = 0; i < BDIM; ++i) {
    A_Array[i] = A + i * MDIM * KDIM;
    B_Array[i] = B + i * KDIM * NDIM;
    C_Array[i] = C + i * MDIM * NDIM;
  }
}

#if defined(CUBLAS)
  cublasHandle_t handle;
  dtype *AA, *BB, *CC;
  const dtype **AA_Array, **BB_Array;
  dtype **CC_Array;
  dtype *devA[BDIM];
  dtype *devB[BDIM];
  dtype *devC[BDIM];
  CHECK_CUBLAS(cublasCreate(&handle));
#ifdef USE_FP16
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  printf("Using FP16 \n");
#else
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  printf("Using FP32 \n");
#endif
  if (BDIM == 0){
    CHECK_CUDA(cudaMalloc((void **)(&AA), MDIM * KDIM * sizeof(dtype)));
    CHECK_CUDA(cudaMalloc((void **)(&BB), KDIM * NDIM * sizeof(dtype)));
    CHECK_CUDA(cudaMalloc((void **)(&CC), MDIM * NDIM * sizeof(dtype)));
    CHECK_CUBLAS(cublasSetVector(MDIM * KDIM, sizeof(dtype), A, 1, AA, 1));
    CHECK_CUBLAS(cublasSetVector(KDIM * NDIM, sizeof(dtype), B, 1, BB, 1));
    CHECK_CUBLAS(cublasSetVector(MDIM * NDIM, sizeof(dtype), C, 1, CC, 1));
  }
  else{
    for (int i = 0; i < BDIM; ++i) {
      CHECK_CUDA(cudaMalloc((void **)(&devA[i]), MDIM * KDIM * sizeof(dtype)));
      CHECK_CUDA(cudaMalloc((void **)(&devB[i]), KDIM * NDIM * sizeof(dtype)));
      CHECK_CUDA(cudaMalloc((void **)(&devC[i]), MDIM * NDIM * sizeof(dtype)));
    }
    CHECK_CUDA(cudaMalloc((void **)(&AA_Array), BDIM * sizeof(dtype*)));
    CHECK_CUDA(cudaMalloc((void **)(&BB_Array), BDIM * sizeof(dtype*)));
    CHECK_CUDA(cudaMalloc((void **)(&CC_Array), BDIM * sizeof(dtype*)));

    for (int i = 0; i < BDIM; i++){
      CHECK_CUDA(cudaMemcpy(devA[i], A_Array[i], MDIM * KDIM * sizeof(dtype), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(devB[i], B_Array[i], KDIM * NDIM * sizeof(dtype), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(devC[i], C_Array[i], MDIM * NDIM * sizeof(dtype), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaMemcpy(AA_Array, devA, BDIM * sizeof(dtype *), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(BB_Array, devB, BDIM * sizeof(dtype *), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(CC_Array, devC, BDIM * sizeof(dtype *), cudaMemcpyHostToDevice));
  }
#endif

#if defined(MLIR_CUDA)
  float *devA, *devB, *devC;
  CHECK_CUDA(cudaMalloc((void **)(&devA), MDIM * KDIM * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)(&devB), KDIM * NDIM * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)(&devC), MDIM * NDIM * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(devA, A, MDIM * KDIM * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(devB, B, KDIM * NDIM * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(devC, C, MDIM * NDIM * sizeof(float), cudaMemcpyHostToDevice));
#endif

#if defined(TVM)
#if defined(USE_TVM_TUNED)
  std::cout << "Using tuned module " << TO_STRING(TVM_LIB) << std::endl;
  tvm::runtime::Module module = tvm::runtime::Module::LoadFromFile(TO_STRING(TVM_LIB));
  assert(module != nullptr);
  tvm::runtime::PackedFunc matmul = module->GetFunction("matmul");
  assert(matmul != nullptr);
#else
  std::cout << "Using baseline module " << std::endl;
  auto module = create_module();
  tvm::runtime::PackedFunc matmul = module->GetFunction("matmul");
#endif

#if defined(TVM_ENABLE_CUDA)
  int deviceType = kDLCUDA;
#elif defined(TVM_ENABLE_METAL)
  int deviceType = kDLMetal;
#elif defined(TVM_ENABLE_ROCM)
  int deviceType = kDLROCM;
#else
  int deviceType = kDLCPU;
#endif
  DLTensor *x, *y, *z;
  int64_t xshape[2] = {MDIM, KDIM};
  TVMArrayAlloc(xshape, 2, kDLFloat, 32, 1, deviceType, 0, &x);
  TVMArrayCopyFromBytes(x, A, MDIM * KDIM * sizeof(float));
  int64_t yshape[2] = {KDIM, NDIM};
  TVMArrayAlloc(yshape, 2, kDLFloat, 32, 1, deviceType, 0, &y);
  TVMArrayCopyFromBytes(y, B, KDIM * NDIM * sizeof(float));
  int64_t zshape[2] = {MDIM, NDIM};
  TVMArrayAlloc(zshape, 2, kDLFloat, 32, 1, deviceType, 0, &z);
#endif

#if defined(COLUMN_MAJOR)
  int LDA = MDIM;
  int LDB = KDIM;
  int LDC = MDIM;
#else
  int LDA = KDIM;
  int LDB = NDIM;
  int LDC = NDIM;
#endif
  float alpha = 1.0;
  float beta = 0.0;

#ifdef RUY
  ruy::Context context;
  context.set_max_num_threads(1);
  ruy::Matrix<float> lhs;
  ruy::MakeSimpleLayout(MDIM, KDIM, MATRIX_FORMAT, lhs.mutable_layout());
  lhs.set_data(A);
  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(KDIM, NDIM, MATRIX_FORMAT, rhs.mutable_layout());
  rhs.set_data(B);
  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(MDIM, NDIM, MATRIX_FORMAT, dst.mutable_layout());
  dst.set_data(C);
  ruy::MulParams<float, float> mul_params;
  lhs.set_cache_policy(ruy::CachePolicy::kCacheIfSignificantSpeedup);
#endif

  for (auto _ : state){
#if defined(MKL) || defined(OPENBLAS) || defined(BLIS) || defined(ACCELERATE)
  if (BDIM == 0){
    cblas_sgemm(MATRIX_FORMAT, CblasNoTrans, CblasNoTrans, MDIM, NDIM, KDIM, alpha,
                A, LDA, B, LDB, beta, C, LDC);
  }
  else{
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    int mdim = MDIM;
    int ndim = NDIM;
    int kdim = KDIM;
    int bdim = BDIM;
    cblas_sgemm_batch(MATRIX_FORMAT, &transA, &transB, &mdim, &ndim, &kdim, &alpha,
                      A_Array, &LDA, B_Array, &LDB, &beta, C_Array, &LDC, 1, &bdim);
  }

#elif defined(HALIDE)
#if defined(COLUMN_MAJOR)
  hblas_sgemm(MATRIX_FORMAT, HblasNoTrans, HblasNoTrans, MDIM, NDIM, KDIM, alpha,
              A, LDA, B, LDB, beta, C, LDC);
#else
  // We assume A, B are row-major (_rm)
  // Since hblas_sgemm only does column major (_cm), we do
  // C_rm = C_cm'
  //      = hblas_sgemm(A_cm, B_cm)'
  //      = hblas_sgemm(B_cm', A_cm')
  //      = hblas_sgemm(B_rm, A_rm)
  hblas_sgemm(MATRIX_FORMAT, HblasNoTrans, HblasNoTrans, NDIM, MDIM, KDIM, alpha,
              B, LDB, A, LDA, beta, C, LDC);
#endif
#elif defined(BLASFEO)
  char c_t = 't';
  int m0 = MDIM;
  int n0 = NDIM;
  int k0 = KDIM;
  blas_sgemm(&c_t, &c_t, &m0, &n0, &k0, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC);
#elif defined(RUY)
  ruy::Mul(lhs, rhs, mul_params, &context, &dst);
#elif defined(TVM)
  matmul(x, y, z);
#elif defined(MLIR) || defined(MLIR_CUDA)
if (BDIM == 0){
#ifdef COLUMN_MAJOR
  matmul(A, A, 0, MDIM, KDIM, 1, LDA,
         B, B, 0, KDIM, NDIM, 1, LDB,
         C, C, 0, MDIM, NDIM, 1, LDC);
#else
  matmul(A, A, 0, MDIM, KDIM, LDA, 1,
         B, B, 0, KDIM, NDIM, LDB, 1,
         C, C, 0, MDIM, NDIM, LDC, 1);
#endif
}
#elif defined(NAIVE)
  naive_matmul(A,B,C,MDIM,KDIM,NDIM);

#elif defined(CUBLAS)
#if defined(COLUMN_MAJOR)
  #ifdef USE_FP16
    if (BDIM == 0){
      CHECK_CUBLAS(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, MDIM, NDIM, KDIM,
                   &alpha, AA, CUDA_R_16F, LDA, BB, CUDA_R_16F, LDB, &beta, CC, CUDA_R_16F, LDC));
    } else {
      CHECK_CUBLAS(cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, MDIM, NDIM, KDIM,
                   &alpha, (void **)&AA_Array[0], CUDA_R_16F, LDA, (void **)&BB_Array[0], CUDA_R_16F, LDB,
                   &beta, (void **)&CC_Array[0], CUDA_R_16F, LDC, BDIM, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
  #else
    if (BDIM == 0){
      CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, MDIM, NDIM, KDIM,
                   &alpha, AA, LDA, BB, LDB, &beta, CC, LDC));
    } else {
      CHECK_CUBLAS(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, MDIM, NDIM, KDIM,
                   &alpha, AA_Array, LDA, BB_Array, LDB, &beta, CC_Array, LDC, BDIM));
    }
  #endif
#else
  #ifdef USE_FP16
    if (BDIM == 0){
      CHECK_CUBLAS(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, NDIM, MDIM, KDIM,
                 &alpha, BB, CUDA_R_16F, LDB, AA, CUDA_R_16F, LDA, &beta, CC, CUDA_R_16F, LDC));
    } else {
      CHECK_CUBLAS(cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, NDIM, MDIM, KDIM,
                   &alpha, (void **)&BB_Array[0], CUDA_R_16F, LDB, (void **)&AA_Array[0], CUDA_R_16F, LDA,
                   &beta, (void **)&CC_Array[0], CUDA_R_16F, LDC, BDIM, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
  #else
    if (BDIM == 0){
      CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NDIM, MDIM, KDIM,
                   &alpha, BB, LDB, AA, LDA, &beta, CC, LDC));
    } else {
      CHECK_CUBLAS(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, NDIM, MDIM, KDIM,
                   &alpha, BB_Array, LDB, AA_Array, LDA, &beta, CC_Array, LDC, BDIM));
    }
  #endif
#endif
#endif
}

#if defined(TVM)
  TVMArrayCopyToBytes(z, C, MDIM * NDIM * sizeof(float));
#elif defined(CUBLAS)
  if (BDIM ==0){
    CHECK_CUBLAS(cublasGetVector(MDIM * NDIM, sizeof(dtype), CC, 1, C, 1));
  }
  else{
    for (int i = 0; i < BDIM; i++){
      CHECK_CUDA(cudaMemcpy(C_Array[i], devC[i], MDIM * NDIM * sizeof(dtype), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaFree(devA[i]));
      CHECK_CUDA(cudaFree(devB[i]));
      CHECK_CUDA(cudaFree(devC[i]));
    }
  }
#endif

#ifdef ENABLE_CHECK
  float *C2 = (float *) malloc(MDIM * NDIM * sizeof(float));
  size_t errors = 0;
  naive_matmul(A,B,C2,MDIM,KDIM,NDIM);
  for (size_t i = 0; i < MDIM; i++) {
    for (size_t j = 0; j < NDIM; j++) {
      size_t ci = i + j*MDIM;
      #ifdef USE_FP16
        float C1 = __half2float(C[ci]);
        if (fabs(C1- C2[ci]) > 0.5f) {  // Difference could be large for mixed precision calculation
          fprintf(stderr, "Incorrect result at index %ld,%ld: C=%0.2f C2=%0.2f\n", i, j, C1, C2[ci]);
          errors++;
        }
      #else
        if (fabs(C[ci] - C2[ci]) > 0.1f) {
          fprintf(stderr, "Incorrect result at index %ld,%ld: C=%0.2f C2=%0.2f\n", i, j, C[ci], C2[ci]);
          errors++;
        }
      #endif
    }
  }
  printf("Detected %ld errors.\n", errors);
#endif

#if defined(TVM)
  TVMArrayFree(x);
  TVMArrayFree(y);
  TVMArrayFree(z);
#elif defined(CUBLAS)
if (BDIM == 0){
  CHECK_CUDA(cudaFree(AA));
  CHECK_CUDA(cudaFree(BB));
  CHECK_CUDA(cudaFree(CC));
}else{
  CHECK_CUDA(cudaFree(AA_Array));
  CHECK_CUDA(cudaFree(BB_Array));
  CHECK_CUDA(cudaFree(CC_Array));
}
  CHECK_CUBLAS(cublasDestroy(handle));
#elif defined(MLIR_CUDA)
  CHECK_CUDA(cudaFree(devA));
  CHECK_CUDA(cudaFree(devB));
  CHECK_CUDA(cudaFree(devC));
#endif
#if 0
  // TODO: For the largest 3 matrix sizes in MLIR, this throws a munmap_chunk(): invalid_pointer
  free(A);
  free(B);
  free(C);
#endif
}

int main(int argc, char **argv) {
#ifdef COLUMN_MAJOR
  printf("Matrix-format: Column-Major\n");
#else
  printf("Matrix-format: Row-Major\n");
#endif
#if defined(ACCELERATE)
  printf("Benchmarking Accelerate %d x %d x %d \n", MDIM, NDIM, KDIM);
#elif defined(BLASFEO)
  printf("Benchmarking BLASFEO %d x %d x %d \n", MDIM, NDIM, KDIM);
#elif defined(BLIS)
  printf("Benchmarking BLIS %d x %d x %d \n", MDIM, NDIM, KDIM);
#elif defined(CUBLAS)
  if (BDIM == 0){
    printf("Benchmarking CUBLAS %d x %d x %d \n", MDIM, NDIM, KDIM);
  } else {
    printf("Benchmarking CUBLAS %d x %d x %d x %d \n", BDIM, MDIM, NDIM, KDIM);
  }
#elif defined(HALIDE)
  printf("Benchmarking Halide %d x %d x %d \n", MDIM, NDIM, KDIM);
#elif defined(MKL)
  if (BDIM == 0){
    printf("Benchmarking MKL %d x %d x %d \n", MDIM, NDIM, KDIM);
  } else {
    printf("Benchmarking MKL %d %d x %d x %d \n", BDIM, MDIM, NDIM, KDIM);
  }
#elif defined(MLIR)
  printf("Benchmarking MLIR %d x %d x %d \n", MDIM, NDIM, KDIM);
#elif defined(MLIR_CUDA)
  printf("Benchmarking MLIR CUDA %d x %d x %d \n", MDIM, NDIM, KDIM);
#elif defined(NAIVE)
  printf("Benchmarking Naive C %d x %d x %d \n", MDIM, NDIM, KDIM);
#elif defined(OPENBLAS)
  printf("Benchmarking OpenBLAS %d x %d x %d \n", MDIM, NDIM, KDIM);
#elif defined(RUY)
  printf("Benchmarking Ruy %d x %d x %d \n", MDIM, NDIM, KDIM);
#elif defined(TVM)
  printf("Benchmarking TVM %d x %d x %d \n", MDIM, NDIM, KDIM);
#endif

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RegisterBenchmark("BM_Matmul", [](benchmark::State& state)
                               -> void {BenchmarkFunction(state);})
                               ->MeasureProcessCPUTime()->UseRealTime();
                               //->Iterations(NUM_REPS); //A fixed number of iterations can be set by uncomment this
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
