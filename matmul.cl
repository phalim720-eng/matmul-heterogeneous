/*
 * matmul.cl — OpenCL Kernel: Matrix Multiplication (float precision)
 *
 * Intel integrated GPU tidak support cl_khr_fp64 (double),
 * sehingga kernel ini menggunakan float (32-bit).
 * Verifikasi dilakukan dengan toleransi 1e-2 untuk mengakomodasi
 * perbedaan rounding float vs double.
 */

__kernel void matmul(
    __global const float *A,
    __global const float *B,
    __global       float *C,
    const int N
) {
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; k++)
        sum += A[row * N + k] * B[k * N + col];

    C[row * N + col] = sum;
}
