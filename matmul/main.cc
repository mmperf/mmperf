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
// TODO: fix the cuMemHostRegister/Unregister problem
// #elif defined(MLIR_CUDA)
// #include "cuda_runtime_api.h"
// #include "cuda.h"
#endif
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <unistd.h>

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

#define CHECK_CUDA(status) do {				        \
  std::stringstream error;					\
  if (status != cudaSuccess) {					\
    printf("Error %d at %s:%d\n", status, __FILE__, __LINE__); 	\
    exit(1);							\
  }								\
} while(0)
#endif

#ifdef TVM
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
  auto target = tvm::Target("llvm");
  auto lowered = tvm::lower(s, args, "matmul", binds);
  auto module = tvm::build(lowered, target, tvm::Target());
  return module;
}
#endif

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void init_matrix(float *a, int nrows, int ncols) {
  for (int j = 0; j < ncols; j++) {
    for (int i = 0; i < nrows; i++) {
      a[i + j * nrows] = ((float) rand() / (float) RAND_MAX);
    }
  }
}

// A print function for sanity check
void print_matrix(float *a, int nrows, int ncols) {
  for (int j = 0; j < ncols; j++) {
    for (int i = 0; i < nrows; i++) {
      printf("%.2f ", a[i + j * nrows]);
    }
    printf("\n");
  }
}

void naive_matmul(const float *a, const float *b, float *c, size_t m, size_t k, size_t n) {
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
        c[ci] += a[i*k + p] * b[p*n + j];
#endif
      }
    }
  }
}

int main(int argc, char **argv) {
#ifdef COLUMN_MAJOR
  printf("Matrix-format: Column-Major\n");
#else
  printf("Matrix-format: Row-Major\n");
#endif
#if defined(ACCELERATE)
  printf("Benchmarking Accelerate %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(BLASFEO)
  printf("Benchmarking BLASFEO %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(BLIS)
  printf("Benchmarking BLIS %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(CUBLAS)
  printf("Benchmarking CUBLAS %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(HALIDE)
  printf("Benchmarking Halide %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(MKL)
  printf("Benchmarking MKL %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(MLIR)
  printf("Benchmarking MLIR %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(MLIR_CUDA)
  printf("Benchmarking MLIR CUDA %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(NAIVE)
  printf("Benchmarking Naive C %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(OPENBLAS)
  printf("Benchmarking OpenBLAS %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(RUY)
  printf("Benchmarking Ruy %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(TVM)
  printf("Benchmarking TVM %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#endif
  double t_start, t_end;
  t_start = rtclock();
  float *A = (float *) malloc(MDIM * KDIM * sizeof(float));
  float *B = (float *) malloc(KDIM * NDIM * sizeof(float));
  float *C = (float *) malloc(MDIM * NDIM * sizeof(float));
  init_matrix(A, MDIM, KDIM);
  init_matrix(B, KDIM, NDIM);
  init_matrix(C, MDIM, NDIM);

// TODO: May be this way cuMemHostRegister problem can be solved,
// but in this case, we need to move gpu.host_register outside of the
// loop.
// #if defined(MLIR_CUDA)
//   float *DA, *DB, *DC;
//   cudaMalloc((void **)&DA, MDIM * KDIM * sizeof(float));
//   cudaMalloc((void **)&DB, MDIM * KDIM * sizeof(float));
//   cudaMalloc((void **)&DC, MDIM * KDIM * sizeof(float));
//
//   cudaMemcpy(DA, A, MDIM * KDIM * sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(DB, B, MDIM * KDIM * sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(DC, C, MDIM * KDIM * sizeof(float), cudaMemcpyHostToDevice);
// #endif

#if defined(CUBLAS)
  cublasHandle_t handle;
  float *AA, *BB, *CC;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUDA(cudaMalloc((void **)(&AA), MDIM * KDIM * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)(&BB), KDIM * NDIM * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)(&CC), MDIM * NDIM * sizeof(float)));
  CHECK_CUBLAS(cublasSetVector(MDIM * KDIM, sizeof(float), A, 1, AA, 1));
  CHECK_CUBLAS(cublasSetVector(KDIM * NDIM, sizeof(float), B, 1, BB, 1));
  CHECK_CUBLAS(cublasSetVector(MDIM * NDIM, sizeof(float), C, 1, CC, 1));
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
#if defined(TVM_ENABLE_CUDA) || defined(TVM_ENABLE_METAL) || defined(TVM_ENABLE_ROCM)
  int deviceType = kDLGPU;
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

  for (int t = 0; t < NUM_REPS; ++t) {
#if defined(MKL) || defined(OPENBLAS) || defined(BLIS) || defined(ACCELERATE)
    cblas_sgemm(MATRIX_FORMAT, CblasNoTrans, CblasNoTrans, MDIM, NDIM, KDIM, alpha,
                A, LDA, B, LDB, beta, C, LDC);
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
#ifdef COLUMN_MAJOR
    matmul(A, A, 0, MDIM, KDIM, 1, LDA,
           B, B, 0, KDIM, NDIM, 1, LDB,
           C, C, 0, MDIM, NDIM, 1, LDC);
#else
    matmul(A, A, 0, MDIM, KDIM, LDA, 1,
           B, B, 0, KDIM, NDIM, LDB, 1,
           C, C, 0, MDIM, NDIM, LDC, 1);
#endif

// TODO: Similar attempt to solve host register problem.
// #elif defined(MLIR_CUDA)
// #ifdef COLUMN_MAJOR
//     matmul(DA, DA, 0, MDIM, KDIM, 1, LDA,
//            DB, DB, 0, KDIM, NDIM, 1, LDB,
//            DC, DC, 0, MDIM, NDIM, 1, LDC);
// #else
//     matmul(DA, DA, 0, MDIM, KDIM, LDA, 1,
//            DB, DB, 0, KDIM, NDIM, LDB, 1,
//            DC, DC, 0, MDIM, NDIM, LDC, 1);
// #endif

#elif defined(NAIVE)
    naive_matmul(A,B,C,MDIM,KDIM,NDIM);
#elif defined(CUBLAS)
#if defined(COLUMN_MAJOR)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, MDIM, NDIM, KDIM,
			     &alpha, AA, LDA, BB, LDB, &beta, CC, LDC));
#else
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NDIM, MDIM, KDIM,
			     &alpha, BB, LDB, AA, LDA, &beta, CC, LDC));
#endif
#endif
  }
  t_end = rtclock();

  int return_code = 0;

#if defined(TVM)
  TVMArrayCopyToBytes(z, C, MDIM * NDIM * sizeof(float));
#elif defined(CUBLAS)
  CHECK_CUBLAS(cublasGetVector(MDIM * NDIM, sizeof(float), CC, 1, C, 1));
#endif

// print result here
// print_matrix(C, MDIM, NDIM);

#ifdef ENABLE_CHECK
  float *C2 = (float *) malloc(MDIM * NDIM * sizeof(float));
  size_t errors = 0;
  naive_matmul(A,B,C2,MDIM,KDIM,NDIM);
  for (size_t i = 0; i < MDIM; i++) {
    for (size_t j = 0; j < NDIM; j++) {
      size_t ci = i + j*MDIM;
      if (std::abs(C[ci] - C2[ci]) > 0.01f) {
        //fprintf(stderr, "Incorrect result at index %ld,%ld: C=%0.2f C2=%0.2f\n", i, j, C[ci], C2[ci]);
        errors++;
      }
    }
  }
  printf("Detected %ld errors.\n", errors);
  if (errors > 0) {
    return_code = 1;
  }
#endif

  const char *filename = TO_STRING(FILE_NAME);
  FILE *file = fopen(filename, "w");
  fprintf(file, "%0.2lf GFLOPS\n", 2.0 * NUM_REPS * MDIM * NDIM * KDIM / (t_end - t_start) / 1E9);
  fclose(file);
#if 0
  // TODO: For the largest 3 matrix sizes in MLIR, this throws a munmap_chunk(): invalid_pointer
  free(A);
  free(B);
  free(C);
#endif

// TODO: We will need this if we use cudaMalloc.
// #if defined(MLIR_CUDA)
//     cudaFree(DA);
//     cudaFree(DB);
//     cudaFree(DC);
// #endif

#if defined(TVM)
  TVMArrayFree(x);
  TVMArrayFree(y);
  TVMArrayFree(z);
#elif defined(CUBLAS)
  CHECK_CUDA(cudaFree(AA));
  CHECK_CUDA(cudaFree(BB));
  CHECK_CUDA(cudaFree(CC));
  CHECK_CUBLAS(cublasDestroy(handle));
#endif
  return return_code;
}
