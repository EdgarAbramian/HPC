#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <cctype>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

struct Image {
    int width, height;
    std::vector<unsigned char> data;
};

// Utility to skip comments in PGM files
void skipComments(std::ifstream& file) {
    char ch;
    while (file.get(ch)) {
        if (ch == '#') {
            std::string dummy;
            std::getline(file, dummy);
        } else if (!isspace(ch)) {
            file.unget();
            break;
        }
    }
}

// Load binary PGM (P5) image
Image readPGM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string magic;
    int width, height, max_val;
    file >> magic;
    if (magic != "P5") {
        std::cerr << "Error: Only binary PGM (P5) supported." << std::endl;
        exit(EXIT_FAILURE);
    }
    skipComments(file);
    file >> width;
    skipComments(file);
    file >> height;
    skipComments(file);
    file >> max_val;
    file.ignore(256, '\n');

    Image img{width, height, std::vector<unsigned char>(width * height)};
    file.read(reinterpret_cast<char*>(img.data.data()), width * height);
    return img;
}

// Save binary PGM (P5) image
void writePGM(const std::string& filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.width * img.height);
}

// GPU kernel for bilinear interpolation using hardware texture filtering
__global__ void bilinear_kernel(cudaTextureObject_t tex, unsigned char* output, int outWidth, int outHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outWidth && y < outHeight) {
        // Map output pixel (x, y) to input coordinates.
        float u = (x + 0.5f) / 2.0f;
        float v = (y + 0.5f) / 2.0f;
        
        // Use tex2D<float> because linear filtering requires floating-point return type.
        // cudaReadModeNormalizedFloat maps [0, 255] to [0.0, 1.0].
        float val = tex2D<float>(tex, u, v);
        output[y * outWidth + x] = (unsigned char)(val * 255.0f + 0.5f);
    }
}

// CPU implementation of bilinear interpolation with wrapping
void bilinear_cpu(const Image& in, Image& out) {
    int inW = in.width;
    int inH = in.height;
    int outW = out.width;
    int outH = out.height;

    for (int y = 0; y < outH; ++y) {
        for (int x = 0; x < outW; ++x) {
            float u = (x + 0.5f) / 2.0f;
            float v = (y + 0.5f) / 2.0f;

            // Find the four surrounding pixels with wrapping logic
            int x0 = floor(u - 0.5f);
            int y0 = floor(v - 0.5f);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float du = u - (x0 + 0.5f);
            float dv = v - (y0 + 0.5f);

            // Apply wrapping (equivalent to cudaAddressModeWrap)
            auto wrap = [](int i, int limit) {
                return (i % limit + limit) % limit;
            };

            int ix0 = wrap(x0, inW);
            int ix1 = wrap(x1, inW);
            int iy0 = wrap(y0, inH);
            int iy1 = wrap(y1, inH);

            float p00 = in.data[iy0 * inW + ix0];
            float p10 = in.data[iy0 * inW + ix1];
            float p01 = in.data[iy1 * inW + ix0];
            float p11 = in.data[iy1 * inW + ix1];

            // Bilinear interpolation formula
            float res = (1.0f - du) * (1.0f - dv) * p00 +
                        du * (1.0f - dv) * p10 +
                        (1.0f - du) * dv * p01 +
                        du * dv * p11;

            out.data[y * outW + x] = (unsigned char)(res + 0.5f);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.pgm> <output.pgm>" << std::endl;
        return 1;
    }

    // 1. Load input image
    Image input = readPGM(argv[1]);
    int outW = input.width * 2;
    int outH = input.height * 2;
    Image output_cpu{outW, outH, std::vector<unsigned char>(outW * outH)};
    Image output_gpu{outW, outH, std::vector<unsigned char>(outW * outH)};

    std::cout << "Image size: " << input.width << "x" << input.height 
              << " -> " << outW << "x" << outH << std::endl;

    // 2. CPU Benchmark
    auto start_cpu = std::chrono::high_resolution_clock::now();
    bilinear_cpu(input, output_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto dur_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();

    // 3. GPU Implementation
    // Allocate and copy data to CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    cudaArray_t cuArray;
    CHECK_CUDA(cudaMallocArray(&cuArray, &channelDesc, input.width, input.height));
    CHECK_CUDA(cudaMemcpy2DToArray(cuArray, 0, 0, input.data.data(), input.width, input.width, input.height, cudaMemcpyHostToDevice));

    // Define texture resource and description
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear; // Enables hardware bilinear interpolation
    texDesc.readMode = cudaReadModeNormalizedFloat; // Required for linear filtering on byte data
    texDesc.normalizedCoords = false;

    // Create Texture Object
    cudaTextureObject_t texObj = 0;
    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    unsigned char* d_out;
    size_t outSize = outW * outH;
    CHECK_CUDA(cudaMalloc(&d_out, outSize));

    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);

    // Warm up
    bilinear_kernel<<<grid, block>>>(texObj, d_out, outW, outH);
    cudaDeviceSynchronize();

    // GPU Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    bilinear_kernel<<<grid, block>>>(texObj, d_out, outW, outH);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float dur_gpu = 0;
    cudaEventElapsedTime(&dur_gpu, start, stop);

    // Copy results back
    CHECK_CUDA(cudaMemcpy(output_gpu.data.data(), d_out, outSize, cudaMemcpyDeviceToHost));

    // 4. Comparison and Results
    int mismatches = 0;
    for (int i = 0; i < outSize; ++i) {
        // Allow +/- 1 difference due to hardware fixed-point interpolation precision
        if (std::abs((int)output_cpu.data[i] - (int)output_gpu.data[i]) > 1) {
            mismatches++;
        }
    }

    std::cout << "CPU Time: " << dur_cpu << " ms" << std::endl;
    std::cout << "GPU Time: " << dur_gpu << " ms" << std::endl;
    std::cout << "Mismatches: " << mismatches << " (within tolerance)" << std::endl;

    // 5. Save output
    writePGM(argv[2], output_gpu);

    // Cleanup
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
