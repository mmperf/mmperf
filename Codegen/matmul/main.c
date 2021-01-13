#ifdef MKL
#include "mkl.h"
#endif
#include "stdio.h"
#include "stdlib.h"
#include <sys/time.h>
#include <unistd.h>

#define STRING(s) #s
#define TO_STRING(x) STRING(x)

struct memref_t {
  double *aligned;
  double *allocated;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
};

void matmul(double *aligned_a, double *allocated_a, int64_t offset_a,
            int64_t size_a0, int64_t size_a1, int64_t strides_a0, int64_t strides_a1,
            double *aligned_b, double *allocated_b, int64_t offset_b,
            int64_t size_b0, int64_t size_b1, int64_t strides_b0, int64_t strides_b1,
            double *aligned_c, double *allocated_c, int64_t offset_c,
            int64_t size_c0, int64_t size_c1, int64_t strides_c0, int64_t strides_c1);

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void init_matrix(double *a, int nrows, int ncols) {
  for (int j = 0; j < ncols; j++) {
    for (int i = 0; i < nrows; i++) {
      a[i + j * nrows] = ((double) rand()) / RAND_MAX;
    }
  }
}

int main(int argc, char **argv) {
#ifdef MKL
  printf("Benchmarking MKL %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#else
  printf("Benchmarking MLIR %d x %d x %d [%d times] \n", MDIM, NDIM, KDIM, NUM_REPS);
#endif
  double t_start, t_end;
  t_start = rtclock();
  double *A = (double *) malloc(MDIM * KDIM * sizeof(double));
  double *B = (double *) malloc(KDIM * NDIM * sizeof(double));
  double *C = (double *) malloc(MDIM * NDIM * sizeof(double));
  init_matrix(A, MDIM, KDIM);
  init_matrix(B, KDIM, NDIM);
  init_matrix(C, MDIM, NDIM);

  int LDA = MDIM;
  int LDB = KDIM;
  int LDC = MDIM;
  double alpha = 1.0;
  double beta = 1.0;

  for (int t = 0; t < NUM_REPS; ++t) {
#ifdef MKL
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, MDIM, NDIM, KDIM, alpha,
                A, LDA, B, LDB, beta, C, LDC);
#else
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
  free(A);
  free(B);
  free(C);
  return 0;
}
