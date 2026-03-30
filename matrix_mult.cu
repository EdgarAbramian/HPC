#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// kernel func for GPU matrix mul
__global__ void matrixMultKernel(const float* A, const float* B, float* C, 
                                  int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// mult using CPU 
void matrixMultCPU(const float* A, const float* B, float* C, 
                   int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; k++) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }
}

// wrapper for GPU clacs
void matrixMultGPU(const float* A, const float* B, float* C, 
                   int rowsA, int colsA, int colsB) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = rowsA * colsA * sizeof(float);
    size_t sizeB = colsA * colsB * sizeof(float);
    size_t sizeC = rowsA * colsB * sizeof(float);
    
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((colsB + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (rowsA + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matrixMultKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, 
                                                      rowsA, colsA, colsB);
    
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// init matrix
void initMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }
}

// display results
bool verifyResults(const float* C_cpu, const float* C_gpu, int rows, int cols) {
    const float epsilon = 1e-3;
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(C_cpu[i] - C_gpu[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <rowsA> <colsA> <colsB>" << std::endl;
        return 1;
    }
    
    int rowsA = atoi(argv[1]);
    int colsA = atoi(argv[2]);
    int colsB = atoi(argv[3]);
    int rowsB = colsA;
    
    srand(time(NULL));
    
    // Выделение памяти
    float* A = new float[rowsA * colsA];
    float* B = new float[rowsB * colsB];
    float* C_cpu = new float[rowsA * colsB];
    float* C_gpu = new float[rowsA * colsB];
    
    initMatrix(A, rowsA, colsA);
    initMatrix(B, rowsB, colsB);
    
    std::cout << "Matrix sizes: A(" << rowsA << "x" << colsA << "), B(" 
              << rowsB << "x" << colsB << ")" << std::endl;
    
    // time check for CPU
    clock_t start_cpu = clock();
    matrixMultCPU(A, B, C_cpu, rowsA, colsA, colsB);
    clock_t end_cpu = clock();
    double cpu_time = static_cast<double>(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    
    // time check for GPU
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    
    cudaEventRecord(start_gpu);
    matrixMultGPU(A, B, C_gpu, rowsA, colsA, colsB);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);
    double gpu_time = gpu_time_ms / 1000.0;
    
    // result validation
    bool correct = verifyResults(C_cpu, C_gpu, rowsA, colsB);
    
    std::cout << "CPU time: " << cpu_time << " s" << std::endl;
    std::cout << "GPU time: " << gpu_time << " s" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;
    std::cout << "Results match: " << (correct ? "YES" : "NO") << std::endl;
    
    // clean used memory blocks 
    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;
    
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    
    return 0;
}