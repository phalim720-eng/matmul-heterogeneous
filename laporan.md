# Laporan Proyek: Heterogeneous Matrix Multiplication

**Mata Kuliah:** Arsitektur dan Sistem Komputer  
**Program:** S1 Kecerdasan Artifisial — Universitas Negeri Surabaya  
**Tahun Akademik:** 2025/2026

---

## 1. Pendahuluan

Perkalian matriks adalah operasi fundamental dalam komputasi ilmiah, machine learning, dan pemrosesan sinyal. Operasi ini memiliki kompleksitas O(N³) sehingga sangat membutuhkan akselerasi paralel untuk ukuran matriks besar.

Proyek ini membandingkan tiga pendekatan:
- **Sequential**: implementasi baseline tanpa paralelisme
- **OpenMP**: CPU parallelism menggunakan shared-memory multi-threading
- **OpenCL**: GPU/accelerator parallelism menggunakan heterogeneous computing model

---

## 2. Arsitektur Sistem

### 2.1 Sequential
Iterasi triple-loop standar dengan optimasi urutan loop (i-k-j) untuk cache locality. Akses memori B dilakukan secara kolom jika tidak di-reorder, sehingga urutan k di luar j meningkatkan hit rate L1/L2 cache.

### 2.2 OpenMP (CPU Parallel)
```
#pragma omp parallel for num_threads(T) schedule(static) shared(A,B,C)
for i = 0..N:       ← dibagi ke T thread
  for k = 0..N:
    for j = 0..N:
      C[i][j] += A[i][k] * B[k][j]
```
Setiap thread menangani subset baris `i`. Tidak ada race condition karena setiap thread menulis ke baris C yang berbeda.

### 2.3 OpenCL (GPU Parallel)
```
Kernel: setiap work-item menghitung SATU elemen C[row][col]
Global work size: N × N work-items
Dimensi: get_global_id(0) = col, get_global_id(1) = row
```
Model memori:
- `__global` A, B, C: berada di global memory device
- Transfer data: host → device (write buffer) → kernel → device → host (read buffer)

---

## 3. Memory Hierarchy

| Level | OpenMP | OpenCL |
|-------|--------|--------|
| Register | variabel lokal loop | sum accumulator per work-item |
| L1/L2 Cache | diakses otomatis oleh CPU | tidak digunakan (no local mem) |
| RAM / Global Mem | shared antar thread | `__global` buffer |
| Transfer | tidak ada (shared mem) | host↔device via PCIe/unified |

**Catatan:** Pada Intel integrated GPU, memori GPU dan CPU bersifat unified (shared physical RAM), sehingga transfer data lebih cepat dibanding dedicated GPU, namun bandwidth tetap terbatas.

---

## 4. Keputusan Desain

| Keputusan | Alasan |
|----------|--------|
| Work-item per elemen (bukan per baris) | Memaksimalkan jumlah work-item paralel di GPU |
| `schedule(static)` di OpenMP | Distribusi beban merata — matriks memiliki ukuran fixed |
| Tidak menggunakan `__local` memory di OpenCL | Menyederhanakan implementasi; local mem tiling meningkatkan kompleksitas tapi signifikan hanya untuk N > 1024 |
| Double precision | Konsistensi dengan referensi sequential untuk verifikasi akurat |
| Warm-up run di OpenCL | JIT compilation kernel terjadi di run pertama; warm-up menghilangkan bias ini dari benchmark |

---

## 5. Hasil Benchmark

*(Isi dengan hasil aktual dari `test/benchmark_results.csv` setelah menjalankan benchmark)*

### 5.1 Tabel Hasil

| Ukuran | Sequential (s) | OpenMP 4T (s) | OpenCL (s) | Speedup OMP | Speedup OCL |
|--------|---------------|---------------|-----------|-------------|-------------|
| 128×128 | — | — | — | — | — |
| 256×256 | — | — | — | — | — |
| 512×512 | — | — | — | — | — |
| 1024×1024 | — | — | — | — | — |

### 5.2 Analisis Speedup

**Mengapa speedup tidak linier (Amdahl's Law):**
- Fraction serial (overhead thread management, fork/join) tidak bisa diparalelkan
- Untuk OpenMP: overhead thread spawn ~0.1–1ms, signifikan untuk N kecil
- Untuk OpenCL: transfer data host↔device, kernel compilation (JIT), dan scheduling overhead

**Mengapa OpenCL pada integrated GPU bisa lebih lambat dari OpenMP:**
- Bandwidth memori unified terbatas (berbagi dengan CPU)
- Intel Iris Plus hanya ~25–40 GB/s bandwidth vs dedicated GPU 200–900 GB/s
- Untuk N kecil, overhead setup OpenCL mendominasi waktu eksekusi

---

## 6. Kelebihan dan Keterbatasan

### Kelebihan
- OpenMP: implementasi sederhana, efektif untuk CPU multicore, tidak ada overhead transfer data
- OpenCL: skalabilitas massif untuk GPU dengan ribuan core (tidak terlihat pada integrated GPU)
- Kode portable: OpenCL berjalan di GPU manapun yang mendukung standar

### Keterbatasan
- OpenCL deprecated di macOS → risiko tidak didukung di masa depan
- Kernel tidak menggunakan local memory tiling → performa sub-optimal untuk N besar
- Intel integrated GPU tidak representatif untuk heterogeneous computing skala besar
- Tidak ada memory pinning (pinned memory) yang bisa meningkatkan transfer bandwidth

---

## 7. Kesimpulan

Parallelisme CPU via OpenMP memberikan speedup yang konsisten dan efisien pada hardware multi-core. OpenCL pada integrated GPU menunjukkan potensi heterogeeneous computing namun dibatasi oleh arsitektur unified memory dan overhead JIT compilation. Untuk problem matrix multiplication dengan N > 2048 pada dedicated GPU, OpenCL diharapkan memberikan speedup yang jauh lebih signifikan.

---

## 8. Referensi

- OpenMP Architecture Review Board. *OpenMP Application Programming Interface Specification*, v5.2, 2021.
- Khronos Group. *OpenCL 1.2 Specification*, 2012.
- Apple Developer Documentation. *OpenCL on Apple Platforms*, 2018.
- Patterson & Hennessy. *Computer Organization and Design*, 5th ed., Morgan Kaufmann.
