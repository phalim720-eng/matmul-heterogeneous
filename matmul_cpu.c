#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

/* ─── Utility ─────────────────────────────────────────── */

/* Allocate flat NxN matrix (row-major) */
double *alloc_matrix(int N) {
    double *m = (double *)malloc(N * N * sizeof(double));
    if (!m) { fprintf(stderr, "malloc failed\n"); exit(1); }
    return m;
}

/* Fill matrix with random values [0, 1) */
void rand_matrix(double *m, int N) {
    for (int i = 0; i < N * N; i++)
        m[i] = (double)rand() / RAND_MAX;
}

/* Elapsed time in seconds using clock_gettime */
double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Verify two result matrices are close enough */
int verify(double *A, double *B, int N, double tol) {
    for (int i = 0; i < N * N; i++) {
        double diff = A[i] - B[i];
        if (diff < 0) diff = -diff;
        if (diff > tol) return 0;
    }
    return 1;
}

/* ─── Sequential ──────────────────────────────────────── */

void matmul_sequential(const double *A, const double *B, double *C, int N) {
    memset(C, 0, N * N * sizeof(double));
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)          /* k-loop hoisted for cache */
            for (int j = 0; j < N; j++)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
}

/* ─── OpenMP Parallel ─────────────────────────────────── */

void matmul_openmp(const double *A, const double *B, double *C, int N, int nthreads) {
    memset(C, 0, N * N * sizeof(double));
    #pragma omp parallel for num_threads(nthreads) schedule(static) shared(A, B, C)
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
}

/* ─── Main ────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    int N        = (argc > 1) ? atoi(argv[1]) : 256;
    int nthreads = (argc > 2) ? atoi(argv[2]) : 4;

    if (N <= 0 || nthreads <= 0) {
        fprintf(stderr, "Usage: %s <matrix_size> <num_threads>\n", argv[0]);
        return 1;
    }

    printf("=== Matrix Multiplication Benchmark (CPU) ===\n");
    printf("Matrix size   : %d x %d\n", N, N);
    printf("Threads (OMP) : %d\n", nthreads);
    printf("Max OMP threads available: %d\n\n", omp_get_max_threads());

    srand(42);
    double *A    = alloc_matrix(N);
    double *B    = alloc_matrix(N);
    double *C_seq = alloc_matrix(N);
    double *C_omp = alloc_matrix(N);

    rand_matrix(A, N);
    rand_matrix(B, N);

    /* --- Sequential --- */
    double t0 = now_sec();
    matmul_sequential(A, B, C_seq, N);
    double t_seq = now_sec() - t0;
    printf("[Sequential]  Time: %.4f s\n", t_seq);

    /* --- OpenMP --- */
    t0 = now_sec();
    matmul_openmp(A, B, C_omp, N, nthreads);
    double t_omp = now_sec() - t0;
    printf("[OpenMP %2d T] Time: %.4f s\n", nthreads, t_omp);

    /* --- Verify --- */
    if (verify(C_seq, C_omp, N, 1e-6))
        printf("\n[Verify] OpenMP result CORRECT ✓\n");
    else
        printf("\n[Verify] OpenMP result MISMATCH ✗\n");

    /* --- Speedup --- */
    double speedup = t_seq / t_omp;
    double efficiency = speedup / nthreads * 100.0;
    printf("\n--- Speedup Analysis ---\n");
    printf("Speedup       : %.2fx\n", speedup);
    printf("Efficiency    : %.1f%%\n", efficiency);
    printf("GFLOPS (seq)  : %.2f\n", 2.0 * N * N * N / t_seq / 1e9);
    printf("GFLOPS (omp)  : %.2f\n", 2.0 * N * N * N / t_omp / 1e9);

    free(A); free(B); free(C_seq); free(C_omp);
    return 0;
}