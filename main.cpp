#include <iostream>
#include "torch/script.h"

int main(){
    torch::jit::script::Module net = torch::jit::load("../models/net.pt"); //"/mnt/d/cpp/net.pt"); //主机本地路径
    torch::Tensor x = torch::randn({1, 100}).to(torch::kCUDA);
    std::cout << "----------第一个输出-------------" << std::endl;
    std::cout << x << std::endl;
    std::vector<torch::jit::IValue> input;
    input.push_back(x);
    for (int i = 0; i < 10; i++){
        std::cout << "----------第" << i+1 << "次前向传播-------------" << std::endl;
        auto out = net.forward(input);
        std::cout << out << std::endl;
    }
    // std::cout << "----------typeid(out)-------------" << std::endl;
    // std::cout << typeid(out).name() << std::endl;
    return 0;
}



// #include <torch/torch.h>
// #include <iostream>

// int main() {
//     // Check if CUDA is available
//     if (torch::cuda::is_available()) {
//         std::cout << "CUDA is available! Training on GPU." << std::endl;
//     } else {
//         std::cout << "CUDA is not available. Training on CPU." << std::endl;
//     }

//     // Create a tensor and perform a simple operation
//     torch::Tensor tensor = torch::rand({2, 3});
//     std::cout << "Random Tensor:" << std::endl << tensor << std::endl;

//     // 在CPU上创建张量
//     torch::Tensor cpu_tensor = torch::randn({3, 3});
//     std::cout << "CPU Tensor:" << std::endl << cpu_tensor << std::endl;
//     std::cout << "CPU Tensor Device: " << cpu_tensor.device() << std::endl;

//     // 在GPU上创建张量（如果CUDA可用）
//     if (torch::cuda::is_available()) {
//         torch::Tensor gpu_tensor = torch::randn({3, 3}, torch::kCUDA);
//         std::cout << "GPU Tensor:" << std::endl << gpu_tensor << std::endl;
//         std::cout << "GPU Tensor Device: " << gpu_tensor.device() << std::endl;
//     }   else {
//         std::cout << "Skipping GPU tensor creation since CUDA is not available." << std::endl;
//     }

//     return 0;
// }