#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include "cuda_runtime.h"
#include <iostream>

__global__ void computeIntensity(const uchar* img1, const uchar* img2, uchar* result, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int offset = y * cols + x;

        float intensity = ((img1[offset * 3] + img1[offset * 3 + 1] + img1[offset * 3 + 2]) +
                          (img2[offset * 3] + img2[offset * 3 + 1] + img2[offset * 3 + 2])) / 6.0f;

        result[offset] = static_cast<uchar>(255 * intensity / 510);
    }
}

int main() {
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	
    cv::Mat image1 = cv::imread("../src/anime1280x960.jpg");
    cv::Mat image2 = cv::imread("../src/gora1280x960.jpg");
    if (image1.empty() || image2.empty()) {
        printf("Images loading error\n");
        return -1;
    }
    std::cout<<"image 1 size: "<<image1.size()<<" image 2 size: " << image2.size()<<"\n";
    for (int iter = 1; iter <= 10; iter++) {
        cv::cuda::GpuMat gpuImage1, gpuImage2, gpuResult;
        gpuImage1.upload(image1);
        gpuImage2.upload(image2);

        gpuResult.create(image1.size(), CV_8UC1);

        const dim3 block(32, 32);
        const dim3 grid((image1.cols + block.x - 1) / block.x, (image1.rows + block.y - 1) / block.y);
        cudaEventRecord(start, 0);
        computeIntensity<<<grid, block>>>(gpuImage1.data, gpuImage2.data, gpuResult.data, image1.rows, image1.cols);
        cudaEventRecord(stop, 0);
        cudaDeviceSynchronize();
        float result_time_cpu;
	    cudaEventElapsedTime(&result_time_cpu, start, stop);
	    printf("ex: %d, time: %f milliseconds\n", iter, result_time_cpu); 
        cv::Mat result;
        gpuResult.download(result);

        cv::imwrite("../res/ex"+std::to_string(iter)+".jpg", result);
    }
    return 0;
}
