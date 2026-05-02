# Harris Corner Detector

A performance comparison between sequential CPU and parallel GPU implementations of the Harris Corner Detector using CUDA.

## Methodology
* **CPU:** Sequential implementation using Sobel operators for gradient calculation, Gaussian weighting for integration, and Non-Maximum Suppression (NMS).
* **GPU:** Parallelized using two optimized CUDA kernels:
    * **Response Kernel:** Utilizes **Texture Memory** (CUDA Texture Objects) for efficient 2D spatial data access during gradient and response calculations.
    * **NMS Kernel:** Parallelized Non-Maximum Suppression to isolate corner peaks above a threshold.

## Results
Performance benchmarked using the provided `img/test.pgm` image.

| Implementation | Execution Time (ms) | Speedup |
| :--- | :--- | :--- |
| CPU (Sequential) | 1144.00 | 1.00x |
| GPU (CUDA) | 94.78 | 12.07x |

**Accuracy:** 
* **Mismatches:** 0 
* Results are bit-perfect compared to the CPU reference implementation.

## Key Highlights
1.  **Memory Optimization:** Leveraging Texture Units provides hardware-accelerated caching for spatial locality, which is ideal for stencil-like operations (Sobel/Gaussian).
2.  **Scalability:** The GPU implementation provides a significant **~12x speedup**, making it suitable for real-time image processing tasks.
3.  **Correctness:** Zero mismatches between CPU and GPU verify the reliability of the parallelized logic.
