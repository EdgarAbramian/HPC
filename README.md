## Lab 3: Bilinear Interpolation
Performance comparison for upscaling a 4K image (3840x2160) to 8K (7680x4320) using hardware-accelerated interpolation.

### Methodology
* **CPU**: Sequential bilinear interpolation calculating weights manually with boundary wrapping.
* **GPU**: Hardware-accelerated interpolation using **CUDA Texture Units** (`cudaFilterModeLinear`). Edge wrapping is handled by `cudaAddressModeWrap`.

### Results
| Implementation | Execution Time (ms) | Speedup |
| :--- | :--- | :--- |
| CPU (Sequential) | 2163.00 | 1.00x |
| GPU (CUDA Hardware) | 0.82 | **~2637x** |

**Accuracy Analysis:**
* **Mismatches:** 18,786 pixels.
* **Explanation:** Discrepancies are within the expected tolerance. They occur due to the difference between GPU hardware fixed-point interpolation and CPU software floating-point calculations.

### Visual Verification
* **Input Image**: `img/test.pgm` (3840x2160)
* **Upscaled Image**: `img/out_bilinear.pgm` (7680x4320)
