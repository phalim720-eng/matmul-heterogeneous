# Heterogeneous Matrix Multiplication
### Arsitektur dan Sistem Komputer — UAS Genap 2025/2026
**Universitas Negeri Surabaya — S1 Kecerdasan Artifisial**

---

## Penyusun Proyek

| Nama | NIM |
|------|-----|
| [Muhamad Halim Pratama] | [25032014016] |
| [Diega Karunia Akbar] | [25032014060] |
| [Brilian Nur Muhammad] | [25032014002] |

---

## Deskripsi Proyek

Proyek ini mengimplementasikan **perkalian matriks (Matrix Multiplication)** menggunakan tiga pendekatan:

| Pendekatan | Teknologi | Target Hardware |
|-----------|-----------|-----------------|
| Sequential | C murni | CPU (1 thread) |
| CPU Parallel | OpenMP | CPU (multi-thread) |
| GPU Parallel | OpenCL | GPU / Integrated GPU |

Tujuan utama adalah membandingkan performa ketiga pendekatan melalui **speedup**, **efisiensi**, dan **GFLOPS** pada berbagai ukuran matriks (128×128 hingga 1024×1024).

### Fitur
- Implementasi sequential sebagai baseline
- Parallelisasi CPU dengan `#pragma omp parallel for` (OpenMP)
- Parallelisasi GPU dengan OpenCL kernel (setiap work-item menghitung satu elemen C)
- Verifikasi kebenaran hasil antara sequential vs parallel
- Benchmark otomatis dengan output CSV
- Kompatibel dengan macOS (Intel) — menggunakan framework OpenCL bawaan Apple

---

## Cara Build & Run (macOS Intel)
cari file di folder : cd Desktop/cpp/matmul-heterogeneous\ 2
cara run CPU : clang -O2 \
  -Xclang -fopenmp \
  -I/usr/local/opt/libomp/include \
  -L/usr/local/opt/libomp/lib \
  -lomp \
  src/matmul_cpu.c -o src/matmul_cpu

./src/matmul_cpu 512 4 (sesuaikan ukuran matriks)

cara run GPU : clang -O2 src/matmul_gpu.c -framework OpenCL -o src/matmul_gpu

KERNEL_PATH=src/matmul.cl ./src/matmul_gpu 512 (sesuaikan ukuran matriks)

### Prasyarat

```bash
# Pastikan Xcode Command Line Tools ter-install
xcode-select --install

# Cek clang dan OpenMP tersedia
clang --version
clang -fopenmp /dev/null -o /dev/null 2>&1 || echo "OpenMP perlu diinstall"

# Install OpenMP via Homebrew jika belum ada
brew install libomp
```

### Build Manual

```bash
# Clone repo
git clone https://github.com/[username]/matmul-heterogeneous.git
cd matmul-heterogeneous

# Build CPU (Sequential + OpenMP)
clang -O2 -fopenmp src/matmul_cpu.c -o src/matmul_cpu

# Build GPU (OpenCL) — macOS
clang -O2 src/matmul_gpu.c -framework OpenCL -o src/matmul_gpu
```

### Run Manual

```bash
# CPU benchmark: ukuran 512x512, 4 thread
./src/matmul_cpu 512 4

# GPU benchmark: ukuran 512x512
KERNEL_PATH=src/matmul.cl ./src/matmul_gpu 512
```

### Run Full Benchmark (otomatis)

```bash
chmod +x scripts/benchmark.sh
bash scripts/benchmark.sh
# Hasil tersimpan di test/benchmark_results.csv
```

---

## Struktur Repository

```
matmul-heterogeneous/
├── src/
│   ├── matmul_cpu.c     # Sequential + OpenMP host code
│   ├── matmul_gpu.c     # OpenCL host code
│   └── matmul.cl        # OpenCL kernel
├── docs/
│   ├── laporan.md       # Laporan singkat
│   ├── flowchart.png    # Diagram arsitektur sistem
│   └── hasil_benchmark/ # Screenshot hasil pengujian
├── test/
│   └── benchmark_results.csv   # Hasil benchmark (di-generate oleh script)
├── scripts/
│   └── benchmark.sh     # Skrip benchmark otomatis
└── README.md
```

---

## 🎥 Video Demo

> 📹 (https://youtu.be/dMK3x51pHRQ)

---

## Hasil Benchmark

*(Dijalankan di MacBook Intel Core i5, Intel Iris Plus Graphics)*

| Ukuran | Sequential | OpenMP (4T) | OpenCL | Speedup OMP | Speedup OCL |
|--------|-----------|-------------|--------|-------------|-------------|
| 128×128 | ~0.0014s | ~0.0028s | ~0.0008s | ~0.48× | ~0.56× |
| 256×256 | ~0.0082s | ~0.0041s | ~0.0026s | ~2.03× | ~1.09× |
| 512×512 | ~0.0510s  | ~0.0188s  | ~0.0175s  | ~2.71× | ~2.03× |
| 1024×1024 | ~0.5983s | ~0.1565s  | ~0.1103s  | ~3.82× | ~1.94× |

---

## Analisis Singkat

- **OpenMP** efektif untuk matriks besar karena distribusi baris antar thread mengurangi serial fraction
- **OpenCL** pada integrated GPU lebih lambat dari dedicated GPU karena berbagi memori dengan CPU (unified memory), sehingga transfer data tidak se-efisien dedicated VRAM
- Speedup tidak linier karena: overhead thread spawning (OpenMP), memory latency, dan Amdahl's Law
- Untuk N kecil (128×128), overhead parallelism mendominasi → sequential lebih cepat

---

## Catatan Teknis

- OpenCL di macOS berstatus **deprecated sejak macOS 10.14**, namun masih fungsional pada Intel Mac
- Kernel menggunakan `double` precision — pastikan device mendukung `cl_khr_fp64` extension
- Jika OpenCL gagal menemukan GPU device, program otomatis fallback ke CPU device
