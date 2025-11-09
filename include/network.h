#pragma once

#include <iostream>
#include <torch/torch.h>

struct NetImpl : torch::nn::Module {
    NetImpl(int fc1_dims, int fc2_dims):
    fc1(fc1_dims, fc1_dims), fc2(fc1_dims, fc2_dims), out(fc2_dims, 1){
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("out", out);
    }

    torch::Tensor forward(torch::Tensor x){
        // 获取模型设备
        torch::Device model_device = this->parameters()[0].device();
        
        // 设备检查（使用 std::cout 替代 LOG(WARNING)）
        if (x.device() != model_device) {
            std::cout << "WARNING: Input tensor device (" << x.device() 
                      << ") doesn't match model device (" << model_device 
                      << "). Automatically moving input." << std::endl;
            x = x.to(model_device);
        }

        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = out(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, out{nullptr};
};

TORCH_MODULE(Net);


  