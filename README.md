# Parallel Matrix Multiplication (CUDA & OpenMP)

High-performance implementation of `C = A × B`, where:
- `A` is an `n × k` matrix
- `B` is a `k × n` matrix
- `C` is the resulting `n × n` product

This repo contains serial, OpenMP, and CUDA-based versions, with performance and accuracy benchmarks.

---

## 🔧 Implementations

### Serial (Baseline)
- Written in C with six loop orderings tested
- 1D row-major arrays for improved memory access

### OpenMP (CPU Parallel)
- `collapse(2)` to spread computation across threads
- SIMD acceleration using `#pragma omp simd`
- Transposes matrix `B` to improve cache locality
- Scales up to 64 threads

### CUDA (GPU Accelerated)
- Tiled matrix multiplication using shared memory
- Coalesced access patterns via Bᵗ layout
- Block sizes tested: 8, 16, 32
- Synchronization for tile-wise collaboration

---

## 📊 Performance Summary

| Method   | Peak GFLOPS | Max Speedup vs Serial |
|----------|-------------|------------------------|
| Serial   | ~0.46       | 1×                     |
| OpenMP   | ~100        | ~30×                  |
| CUDA     | ~3282       | ~300×                 |

---

## ✅ Accuracy Check

All outputs validated against the serial version using:
max(abs(C_parallel - C_serial)) ≤ 1e-5
Results exceeding the threshold are flagged in red in the figures.

---

## 📈 CUDA Benchmark (Tesla V100)

<img src="CUDA Matrix Multiplication Performance Validation.png" alt="CUDA Performance" width="100%">

- Block size 32 peaked at **3282 GFLOPS**
- Most large matrix sizes and `k` values pass validation
- Performance improves with block size and dimensions

---

## 📈 OpenMP Benchmark (CPU 64 Threads)

<img src="Openmp Matrix Multiplication Performance Validation.png" alt="OpenMP Performance" width="100%">

- Reached up to **100 GFLOPS** on 64 threads
- Precision drops slightly at high `k` or low `n`
- Performance increases steadily with thread count

---

## 🛠 Optimization Techniques

| Component     | Optimization                          |
|---------------|----------------------------------------|
| Loop Order    | Best found: `I-L-J`, `L-I-J`           |
| SIMD          | Manual vectorization in inner loop     |
| Memory Layout | Transposed `B` for OpenMP              |
| Shared Memory | Used in CUDA for tiling                |
| Coalesced Access | Improved CUDA memory throughput     |
