#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

// sum on cpu
float sumCPU(const float* arr, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
    return sum;
}

// CUDA kernel 
__global__ void sumReductionKernel(const float* arr, float* out, int size) {
    // shared memory
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // loading data
    sdata[tid] = (i < size) ? arr[i] : 0.0f;
    __syncthreads(); // sync until all threads will gettin thir data chunks

    // reduction with shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // waite for each sum 
    }

    // master-thread (th with num 0) get all sums from other threads
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

float sumGPU(const float* d_arr, int size, float* time_ms) {
    float* d_out;
    cudaMalloc((void**)&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float)); // Инициализируем сумму нулем

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    sumReductionKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_arr, d_out, size);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(time_ms, start, stop);

    // copy results on host device
    float result = 0.0f;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

void runExperiment(int size) {
    // Get memory for CPU execution
    float* h_arr = new float[size];
    for (int i = 0; i < size; ++i) {
        h_arr[i] = 1.0f; // fill vec with 1 for test
    }

    float* d_arr;
    cudaMalloc((void**)&d_arr, size * sizeof(float));
    cudaMemcpy(d_arr, h_arr, size * sizeof(float), cudaMemcpyHostToDevice);

    // CPU Test 
    auto startCPU = std::chrono::high_resolution_clock::now();
    float cpuResult = sumCPU(h_arr, size);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpuDuration = endCPU - startCPU;
    float cpuTimeMs = cpuDuration.count();

    // GPU Test
    float gpuTimeMs = 0.0f;
    float gpuResult = sumGPU(d_arr, size, &gpuTimeMs);

    // Calc Acceleration
    float acceleration = cpuTimeMs / gpuTimeMs;

    std::cout << std::left << std::setw(12) << size 
              << std::setw(15) << cpuTimeMs 
              << std::setw(15) << gpuTimeMs 
              << std::setw(15) << acceleration 
              << "(CPU Sum: " << cpuResult << ", GPU Sum: " << gpuResult << ")\n";

    // free memory that we used above
    delete[] h_arr;
    cudaFree(d_arr);
}

int main() {
    std::cout << std::left << std::setw(12) << "Array Size" 
              << std::setw(15) << "CPU Time (ms)" 
              << std::setw(15) << "GPU Time (ms)" 
              << std::setw(15) << "Acceleration" << "\n";

    int sizes[] = {1000, 10000, 50000, 100000, 500000, 1000000};
    for (int size : sizes) {
        runExperiment(size);
    }

    return 0;
}