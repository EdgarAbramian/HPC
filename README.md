# Lab 1: Vector Sum

A simple performance comparison between sequential CPU summation and parallel GPU reduction using CUDA.

## Methodology
* **CPU:** A single-thread `for` loop over a dynamic array.
* **GPU:** Parallel reduction using **Shared Memory**. The array is divided into blocks (256 threads each) to calculate partial sums, followed by an `atomicAdd` for the final result. This minimizes global memory bottlenecks.

## Results
Tests performed on a Google Colab T4 GPU.

| Array Size | CPU Time (ms) | GPU Time (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| 1,000 | 0.0032 | 0.1802 | 0.02x |
| 10,000 | 0.0302 | 0.0146 | 2.06x |
| 50,000 | 0.1526 | 0.0307 | 4.97x |
| 100,000 | 0.3171 | 0.0207 | 15.34x |
| 500,000 | 1.4502 | 0.0519 | 27.92x |
| 1,000,000 | 3.1918 | 0.0942 | 33.89x |



## Key Findings
1.  **Overhead:** For small datasets ($N < 10^4$), the CPU wins because GPU kernel invocation and memory transfer costs too much raher than the calculation time.
2.  **Scalability:** As the workload increases, the GPU's massive parallelism takes over, achieving a **~34x speedup** at 1 million elements.
