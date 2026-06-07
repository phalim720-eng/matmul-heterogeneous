#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
float *alloc_matrix(int N) {
    float *m = (float *)malloc(N * N * sizeof(float));
    if (!m) { fprintf(stderr, "malloc failed\n"); exit(1); }
    return m;
}
void rand_matrix(float *m, int N) {
    for (int i = 0; i < N * N; i++) m[i] = (float)rand() / RAND_MAX;
}
void matmul_ref(const float *A, const float *B, float *C, int N) {
    memset(C, 0, N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
}
int verify(const float *ref, const float *gpu, int N, float tol) {
    for (int i = 0; i < N * N; i++) {
        float d = fabsf(ref[i] - gpu[i]);
        if (d > tol) {
            printf("[Verify] MISMATCH at index %d: ref=%.4f gpu=%.4f diff=%.2e\n", i, ref[i], gpu[i], d);
            return 0;
        }
    }
    return 1;
}
char *load_kernel_source(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
    char *src = (char *)malloc(len + 1);
    fread(src, 1, len, f); src[len] = '\0'; fclose(f);
    *out_len = (size_t)len;
    return src;
}
#define CL_CHECK(err, msg) \
    do { if ((err) != CL_SUCCESS) { fprintf(stderr, "OpenCL error %d at %s\n", (int)(err), (msg)); exit(1); } } while(0)

int main(int argc, char *argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    const char *kernel_path = getenv("KERNEL_PATH");
    char kpath_buf[1024];
    if (!kernel_path) { snprintf(kpath_buf, sizeof(kpath_buf), "src/matmul.cl"); kernel_path = kpath_buf; }

    printf("=== Matrix Multiplication Benchmark (OpenCL GPU) ===\n");
    printf("Matrix size : %d x %d\n", N, N);
    printf("Precision   : float32 (Intel iGPU no fp64)\n");
    printf("Kernel file : %s\n\n", kernel_path);

    srand(42);
    float *h_A = alloc_matrix(N), *h_B = alloc_matrix(N);
    float *h_C = alloc_matrix(N), *h_ref = alloc_matrix(N);
    rand_matrix(h_A, N); rand_matrix(h_B, N);

    cl_int err;
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms = malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);
    printf("Available OpenCL Platforms:\n");
    for (cl_uint p = 0; p < num_platforms; p++) {
        char name[256]; clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(name), name, NULL);
        printf("  [%u] %s\n", p, name);
    }
    cl_platform_id platform = platforms[0]; free(platforms);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        printf("No GPU found, using CPU device.\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }
    CL_CHECK(err, "clGetDeviceIDs");

    char dev_name[256]; clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    cl_ulong gmem; clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL);
    char extensions[2048]={0}; clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
    printf("\nDevice    : %s\nGlobal mem: %.0f MB\nFP64      : %s\n\n",
           dev_name, gmem/1e6, strstr(extensions,"cl_khr_fp64") ? "YES" : "NO - using float32");

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CL_CHECK(err, "ctx");
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err); CL_CHECK(err, "queue");

    size_t src_len; char *src = load_kernel_source(kernel_path, &src_len);
    if (!src) { fprintf(stderr, "Cannot open kernel: %s\n", kernel_path); return 1; }
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char **)&src, &src_len, &err);
    CL_CHECK(err, "createProgram"); free(src);
    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log); free(log); return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "matmul", &err); CL_CHECK(err, "kernel");

    size_t bytes = N * N * sizeof(float);
    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, bytes, h_A, &err); CL_CHECK(err, "bufA");
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, bytes, h_B, &err); CL_CHECK(err, "bufB");
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err); CL_CHECK(err, "bufC");
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    size_t global[2] = {(size_t)N, (size_t)N};
    cl_event ev;
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &ev);
    clWaitForEvents(1, &ev); clReleaseEvent(ev);

    double t0 = now_sec();
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &ev);
    CL_CHECK(err, "enqueue"); clWaitForEvents(1, &ev);
    double t_gpu = now_sec() - t0;
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes, h_C, 0, NULL, NULL);

    printf("Computing CPU reference...\n");
    double t0r = now_sec(); matmul_ref(h_A, h_B, h_ref, N); double t_seq = now_sec() - t0r;

    printf("[Verify] OpenCL result: %s\n\n", verify(h_ref, h_C, N, 0.5f) ? "CORRECT ✓" : "MISMATCH ✗");
    printf("--- Benchmark Results ---\n");
    printf("Sequential time : %.4f s\n", t_seq);
    printf("OpenCL time     : %.4f s\n", t_gpu);
    printf("Speedup         : %.2fx\n",  t_seq/t_gpu);
    printf("GFLOPS (seq)    : %.2f\n",   2.0*N*N*N/t_seq/1e9);
    printf("GFLOPS (OCL)    : %.2f\n",   2.0*N*N*N/t_gpu/1e9);

    clReleaseEvent(ev); clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseKernel(kernel); clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(ctx);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}