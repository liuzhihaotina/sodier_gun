#include <torch/torch.h>
#include <iostream>

int main() {
    // Check if CUDA is available
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
    } else {
        std::cout << "CUDA is not available. Training on CPU." << std::endl;
    }

    // Create a tensor and perform a simple operation
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Random Tensor:" << std::endl << tensor << std::endl;

    // 在CPU上创建张量
    torch::Tensor cpu_tensor = torch::randn({3, 3});
    std::cout << "CPU Tensor:" << std::endl << cpu_tensor << std::endl;
    std::cout << "CPU Tensor Device: " << cpu_tensor.device() << std::endl;

    // 在GPU上创建张量（如果CUDA可用）
    if (torch::cuda::is_available()) {
        torch::Tensor gpu_tensor = torch::randn({3, 3}, torch::kCUDA);
        std::cout << "GPU Tensor:" << std::endl << gpu_tensor << std::endl;
        std::cout << "GPU Tensor Device: " << gpu_tensor.device() << std::endl;
    }   else {
        std::cout << "Skipping GPU tensor creation since CUDA is not available." << std::endl;
    }

    return 0;
}