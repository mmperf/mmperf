#ifdef MKL
#include "mkl.h"
#elif defined(OPENBLAS)
#include "cblas.h"
#elif defined(HALIDE)
#include "halide_blas.h"
#elif defined(RUY)
#include "ruy/ruy.h"
#endif
#include "stdio.h"
#include "stdlib.h"
#include <sys/time.h>
#include <unistd.h>

#define STRING(s) #s
#define TO_STRING(x) STRING(x)

#ifdef MLIR
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

int main(int argc, char **argv) {
#ifdef MKL
  printf("Benchmarking MKL %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(OPENBLAS)
  printf("Benchmarking OpenBLAS %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(HALIDE)
  printf("Benchmarking Halide %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(RUY)
  printf("Benchmarking Ruy %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#elif defined(MLIR)
  printf("Benchmarking MLIR %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#endif
  double t_start, t_end;
  t_start = rtclock();
  float *A = (float *) malloc(MDIM * KDIM * sizeof(float));
  float *B = (float *) malloc(KDIM * NDIM * sizeof(float));
  float *C = (float *) malloc(MDIM * NDIM * sizeof(float));
  init_matrix(A, MDIM, KDIM);
  init_matrix(B, KDIM, NDIM);
  init_matrix(C, MDIM, NDIM);

  int LDA = MDIM;
  int LDB = KDIM;
  int LDC = MDIM;
  float alpha = 1.0;
  float beta = 1.0;

#ifdef RUY
  ruy::Context context;
  context.set_max_num_threads(1);
  ruy::Matrix<float> lhs;
  ruy::MakeSimpleLayout(MDIM, KDIM, ruy::Order::kColMajor, lhs.mutable_layout());
  lhs.set_data(A);
  ruy::Matrix<float> rhs;
  ruy::MakeSimpleLayout(KDIM, NDIM, ruy::Order::kColMajor, rhs.mutable_layout());
  rhs.set_data(B);
  ruy::Matrix<float> dst;
  ruy::MakeSimpleLayout(MDIM, NDIM, ruy::Order::kColMajor, dst.mutable_layout());
  dst.set_data(C);
  ruy::MulParams<float, float> mul_params;
  mul_params.set_bias(C);
#endif

  for (int t = 0; t < NUM_REPS; ++t) {
#if defined(MKL) || defined(OPENBLAS)
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, MDIM, NDIM, KDIM, alpha,
                A, LDA, B, LDB, beta, C, LDC);
#elif defined(HALIDE)
    hblas_sgemm(HblasColMajor, HblasNoTrans, HblasNoTrans, MDIM, NDIM, KDIM, alpha,
                A, LDA, B, LDB, beta, C, LDC);
#elif defined(RUY)
    ruy::Mul(lhs, rhs, mul_params, &context, &dst);
#elif defined(MLIR)
    matmul(A, A, 0, MDIM, KDIM, 1, LDA,
           B, B, 0, KDIM, NDIM, 1, LDB,
	   C, C, 0, MDIM, NDIM, 1, LDC);
#endif
  }
  t_end = rtclock();
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
  return 0;
}
