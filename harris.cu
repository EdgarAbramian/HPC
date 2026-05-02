#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <cctype>
#include <cuda_runtime.h>

// Error checking macro for CUDA API calls
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

void writePGM(const std::string& filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.width * img.height);
}

__global__ void harris_response_kernel(cudaTextureObject_t tex, float* R, int width, int height, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float ix2_sum = 0.0f, iy2_sum = 0.0f, ixy_sum = 0.0f;
    float gaussian[3][3] = { {1.f/16, 2.f/16, 1.f/16}, {2.f/16, 4.f/16, 2.f/16}, {1.f/16, 2.f/16, 1.f/16} };

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;

            // Read from texture memory for hardware-accelerated caching
            float p00 = tex2D<unsigned char>(tex, nx - 1, ny - 1);
            float p02 = tex2D<unsigned char>(tex, nx + 1, ny - 1);
            float p10 = tex2D<unsigned char>(tex, nx - 1, ny);
            float p12 = tex2D<unsigned char>(tex, nx + 1, ny);
            float p20 = tex2D<unsigned char>(tex, nx - 1, ny + 1);
            float p21 = tex2D<unsigned char>(tex, nx,     ny + 1);
            float p22 = tex2D<unsigned char>(tex, nx + 1, ny + 1);

            float Ix = (p02 + 2*p12 + p22) - (p00 + 2*p10 + p20);
            float Iy = (p20 + 2*p21 + p22) - (p00 + 2*p00 + p02); // Simplified Sobel

            float weight = gaussian[dy+1][dx+1];
            ix2_sum += weight * Ix * Ix;
            iy2_sum += weight * Iy * Iy;
            ixy_sum += weight * Ix * Iy;
        }
    }

    float det = (ix2_sum * iy2_sum) - (ixy_sum * ixy_sum);
    float trace = ix2_sum + iy2_sum;
    // Final Harris corner response
    R[y * width + x] = det - alpha * (trace * trace);
}

__global__ void nms_kernel(const float* R, unsigned char* output, int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float val = R[idx];

    // Non-Maximum Suppression (NMS) in a 3x3 window
    if (val > threshold) {
        bool is_max = true;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = max(0, min(width - 1, x + dx));
                int ny = max(0, min(height - 1, y + dy));
                if (R[ny * width + nx] >= val) { is_max = false; break; }
            }
            if (!is_max) break;
        }
        output[idx] = is_max ? 255 : 0;
    } else {
        output[idx] = 0;
    }
}

// Sequential Harris Corner implementation for validation
void harris_cpu(const Image& img, Image& out, float threshold, float alpha) {
    std::vector<float> R(img.width * img.height, 0.0f);
    auto get_p = [&](int x, int y) {
        x = std::max(0, std::min(img.width - 1, x));
        y = std::max(0, std::min(img.height - 1, y));
        return (float)img.data[y * img.width + x];
    };

    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            float ix2 = 0, iy2 = 0, ixy = 0;
            float g[3][3] = { {1.f/16, 2.f/16, 1.f/16}, {2.f/16, 4.f/16, 2.f/16}, {1.f/16, 2.f/16, 1.f/16} };
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = x + dx, ny = y + dy;
                    float Ix = (get_p(nx+1, ny-1) + 2*get_p(nx+1, ny) + get_p(nx+1, ny+1)) - 
                               (get_p(nx-1, ny-1) + 2*get_p(nx-1, ny) + get_p(nx-1, ny+1));
                    float Iy = (get_p(nx-1, ny+1) + 2*get_p(nx, ny+1) + get_p(nx+1, ny+1)) - 
                               (get_p(nx-1, ny-1) + 2*get_p(nx, ny-1) + get_p(nx+1, ny-1));
                    ix2 += g[dy+1][dx+1] * Ix * Ix;
                    iy2 += g[dy+1][dx+1] * Iy * Iy;
                    ixy += g[dy+1][dx+1] * Ix * Iy;
                }
            }
            R[y * img.width + x] = (ix2 * iy2 - ixy * ixy) - alpha * (ix2 + iy2) * (ix2 + iy2);
        }
    }
    for (int i = 0; i < img.width * img.height; ++i) {
        if (R[i] > threshold) {
            int x = i % img.width, y = i / img.width;
            bool m = true;
            for(int dy=-1; dy<=1; ++dy) {
                for(int dx=-1; dx<=1; ++dx) {
                    if(dx==0 && dy==0) continue;
                    int nx = std::max(0, std::min(img.width-1, x+dx)), ny = std::max(0, std::min(img.height-1, y+dy));
                    if(R[ny*img.width+nx] >= R[i]) { m=false; break; }
                }
                if(!m) break;
            }
            out.data[i] = m ? 255 : 0;
        } else out.data[i] = 0;
    }
}

int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "Usage: " << argv[0] << " <in.pgm> <out.pgm> <thresh>\n"; return 1; }

    Image img = readPGM(argv[1]);
    float threshold = std::stof(argv[3]);
    float alpha = 0.04f;

    Image cpu_out{img.width, img.height, std::vector<unsigned char>(img.width * img.height)};
    auto s_cpu = std::chrono::high_resolution_clock::now();
    harris_cpu(img, cpu_out, threshold, alpha);
    auto e_cpu = std::chrono::high_resolution_clock::now();

    // Create Texture Object for efficient image access
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<unsigned char>();
    cudaArray_t cuArr;
    CHECK_CUDA(cudaMallocArray(&cuArr, &ch, img.width, img.height));
    
    CHECK_CUDA(cudaMemcpy2DToArray(cuArr, 0, 0, img.data.data(), img.width * sizeof(unsigned char), 
                                   img.width * sizeof(unsigned char), img.height, cudaMemcpyHostToDevice));

    cudaResourceDesc res = {};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = cuArr;

    cudaTextureDesc tex = {};
    tex.addressMode[0] = tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModePoint;
    tex.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj = 0;
    CHECK_CUDA(cudaCreateTextureObject(&texObj, &res, &tex, nullptr));

    float *d_R;
    unsigned char *d_out;
    CHECK_CUDA(cudaMalloc(&d_R, img.width * img.height * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, img.width * img.height));

    dim3 block(16, 16);
    dim3 grid((img.width + 15) / 16, (img.height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    // Execute GPU implementation
    harris_response_kernel<<<grid, block>>>(texObj, d_R, img.width, img.height, alpha);
    nms_kernel<<<grid, block>>>(d_R, d_out, img.width, img.height, threshold);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    Image gpu_res{img.width, img.height, std::vector<unsigned char>(img.width * img.height)};
    CHECK_CUDA(cudaMemcpy(gpu_res.data.data(), d_out, img.width * img.height, cudaMemcpyDeviceToHost));

    std::cout << "CPU: " << std::chrono::duration_cast<std::chrono::milliseconds>(e_cpu - s_cpu).count() << "ms\n";
    std::cout << "GPU: " << ms << "ms\n";

    int diff = 0;
    for (int i = 0; i < img.width * img.height; ++i) {
        if (cpu_out.data[i] != gpu_res.data[i]) diff++;
        gpu_res.data[i] = (gpu_res.data[i] == 255) ? 255 : img.data[i];
    }
    std::cout << "Mismatches: " << diff << "\n";

    writePGM(argv[2], gpu_res);

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArr);
    cudaFree(d_R); cudaFree(d_out);
    return 0;
}