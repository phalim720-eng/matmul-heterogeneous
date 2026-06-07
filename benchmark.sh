#!/usr/bin/env bash
# benchmark.sh — Run full benchmark across matrix sizes and thread counts
# Usage: bash scripts/benchmark.sh
# Output: test/benchmark_results.csv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SRC="$ROOT_DIR/src"
TEST="$ROOT_DIR/test"

echo "=== Building binaries ==="
cd "$ROOT_DIR"

# Build CPU (OpenMP)
clang -O2 -fopenmp "$SRC/matmul_cpu.c" -o "$SRC/matmul_cpu"
echo "[OK] matmul_cpu built"

# Build GPU (OpenCL) — macOS
clang -O2 "$SRC/matmul_gpu.c" -framework OpenCL -o "$SRC/matmul_gpu"
echo "[OK] matmul_gpu built"

mkdir -p "$TEST"

CSV="$TEST/benchmark_results.csv"
echo "type,size,threads,time_s,speedup_vs_seq,gflops" > "$CSV"

SIZES=(128 256 512 1024)
THREAD_COUNTS=(1 2 4 8)

echo ""
echo "=== Running CPU Benchmarks ==="

for N in "${SIZES[@]}"; do
    echo ""
    echo "--- Matrix size: ${N}x${N} ---"

    # Sequential (1 thread as baseline)
    OUTPUT=$("$SRC/matmul_cpu" "$N" 1 2>&1)
    echo "$OUTPUT"
    T_SEQ=$(echo "$OUTPUT" | grep "Sequential" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    GFLOPS_SEQ=$(echo "$OUTPUT" | grep "GFLOPS (seq)" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo "sequential,$N,1,$T_SEQ,1.00,$GFLOPS_SEQ" >> "$CSV"

    # OpenMP with multiple thread counts
    for T in "${THREAD_COUNTS[@]}"; do
        [ "$T" -eq 1 ] && continue   # already done
        OUTPUT=$("$SRC/matmul_cpu" "$N" "$T" 2>&1)
        T_OMP=$(echo "$OUTPUT" | grep "OpenMP" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        SPEEDUP=$(echo "$OUTPUT" | grep "Speedup" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        GFLOPS=$(echo "$OUTPUT" | grep "GFLOPS (omp)" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        echo "openmp,$N,$T,$T_OMP,$SPEEDUP,$GFLOPS" >> "$CSV"
    done
done

echo ""
echo "=== Running GPU (OpenCL) Benchmarks ==="

for N in "${SIZES[@]}"; do
    echo ""
    echo "--- Matrix size: ${N}x${N} ---"
    OUTPUT=$(KERNEL_PATH="$SRC/matmul.cl" "$SRC/matmul_gpu" "$N" 2>&1)
    echo "$OUTPUT"
    T_GPU=$(echo "$OUTPUT"  | grep "OpenCL time"  | grep -oE '[0-9]+\.[0-9]+' | head -1)
    SPEEDUP=$(echo "$OUTPUT" | grep "Speedup"     | grep -oE '[0-9]+\.[0-9]+' | head -1)
    GFLOPS=$(echo "$OUTPUT"  | grep "GFLOPS (OCL)" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo "opencl,$N,N/A,$T_GPU,$SPEEDUP,$GFLOPS" >> "$CSV"
done

echo ""
echo "=== Benchmark complete ==="
echo "Results saved to: $CSV"
cat "$CSV"
