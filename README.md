# Lab 0: Matrix Multiplication (CPU vs GPU)

This repository contains the implementation of matrix multiplication using two different approaches: a sequential CPU-based method and a parallelized CUDA-based GPU method.

## Project Description
The core of this project is a performance comparison between traditional serial processing and modern parallel computing. 
- **CPU Implementation:** Uses a standard triple-nested loop to calculate the dot product of rows and columns.
- **GPU Implementation:** Uses a custom CUDA kernel (`matrixMultKernel`) that maps each element of the resulting matrix to a specific GPU thread. This allows thousands of calculations to occur simultaneously.
- **Environment:** Due to local hardware constraints, all tests were executed using **Google Colab** with an NVIDIA Tesla T4 GPU.

## Performance Results
The benchmarks show the execution time (in seconds) for various matrix sizes ($N \times N$).

| Size ($N$) | CPU Time (s) | GPU Time (s) | Speedup |
| :--- | :--- | :--- | :--- |
| 100 | 0.003132 | 0.102049 | 0.03x |
| 200 | 0.024870 | 0.000678 | 36.63x |
| 500 | 0.466650 | 0.002386 | 195.52x |
| 800 | 1.905370 | 0.006371 | 299.06x |
| 1000 | 4.930890 | 0.009327 | 528.63x |
| 1200 | 6.594260 | 0.014929 | 441.68x |
| 1500 | 19.34060 | 0.036583 | 528.66x |
| 1800 | 42.19050 | 0.049979 | 844.15x |
| 2000 | 62.47100 | 0.054521 | 1145.8x |