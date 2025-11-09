#include "dataset.h"
#include <iostream>
#include <unordered_map>
#include <torch/torch.h>

int main() {
        // 创建数据集
        auto dataset = MyDataset("data.bin")
            // .map(torch::data::transforms::Normalize<>(0.5, 0.5))  // 数据变换
            .map(torch::data::transforms::Stack<>());             // 堆叠成批次
    
        // 创建数据加载器
        auto dataloader = torch::data::make_data_loader(
                std::move(dataset),
                torch::data::DataLoaderOptions()
                    .batch_size(2)
                    .workers(4)
                    .enforce_ordering(false)  // 是否保持顺序
            );
        
            // 训练循环
            int i = 0;
            for (auto& batch : *dataloader) {
        i++;
        auto data = batch.data;    // 输入数据
        auto target = batch.target; // 目标标签
        
        // 前向传播、损失计算、反向传播...
        std::cout << "------第 " << i << "次------" << std::endl;
        std::cout << "data[0][0][0][0] = " << data[0][0][0][0] << std::endl;
        std::cout << "data[0][0][0] = " << data[0][0][0] << std::endl;
        std::cout << "Batch size: " << data.sizes() << std::endl;
        std::cout << "target size: " << target.sizes() << std::endl;
        std::cout << "target = " << target << std::endl;
        if (i == 1){            
            // 保存张量到文件
            auto tensor_dict;
            tensor_dict["data"] = data;
            tensor_dict["targets"] = target;
            torch::save(tensor_dict, "../pt/tensor_dict.pt");
        }
    }

    return 0;
}

// #include "kitti_dataset/kitti_depth_completion.h"
// #include "kitti_dataset/kitti_depth_prediction.h"
// #include "kitti_dataset/transforms.h"
// #include <torch/torch.h>
// #include <iostream>

// int main() {
//     std::string dataset_path = "/mnt/d/cpp/kitti_dataset";
    
//     try {
//         // 测试深度补全数据集
//         std::cout << "=== Testing Depth Completion Dataset ===" << std::endl;
//         auto completion_dataset = kitti::KITTIDepthCompletionDataset(
//             dataset_path, 
//             kitti::KITTIDataset::VAL
//         ).map(kitti::KITTITransform(false))
//          .map(torch::data::transforms::Stack<>());
        
//         auto completion_loader = torch::data::make_data_loader(
//             std::move(completion_dataset),
//             torch::data::DataLoaderOptions().batch_size(2).workers(1)
//         );
        
//         for (auto& batch : *completion_loader) {
//             std::cout << "Completion batch - Data: " << batch.data.sizes() 
//                       << ", Target: " << batch.target.sizes() << std::endl;
//             break;
//         }
        
//         // 测试深度预测数据集
//         std::cout << "\n=== Testing Depth Prediction Dataset ===" << std::endl;
//         auto prediction_dataset = kitti::KITTIDepthPredictionDataset(
//             dataset_path,
//             kitti::KITTIDataset::TEST_PREDICTION
//         ).map(kitti::KITTITransform(false))
//          .map(torch::data::transforms::Stack<>());
        
//         auto prediction_loader = torch::data::make_data_loader(
//             std::move(prediction_dataset),
//             torch::data::DataLoaderOptions().batch_size(2).workers(1)
//         );
        
//         for (auto& batch : *prediction_loader) {
//             std::cout << "Prediction batch - Data: " << batch.data.sizes() 
//                       << ", Target: " << batch.target.sizes() << std::endl;
//             break;
//         }
        
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }
    
//     return 0;
// }

// #include "network.h"
// #include <iostream>
// #include <torch/torch.h>

// using namespace torch;

// int main(){
//     // 1. 创建网络
//     Net network(50, 10);
    
//     // 2. 将模型移到GPU（如果可用）
//     if (torch::cuda::is_available()) {
//         std::cout << "CUDA available! Using GPU." << std::endl;
//         network->to(torch::kCUDA);
//     } else {
//         std::cout << "Using CPU." << std::endl;
//         network->to(torch::kCPU);
//     }
    
//     std::cout << "Network device: " << network->parameters()[0].device() << std::endl;
//     std::cout << network << "\n\n";
    
//     // 3. 创建输入张量（必须初始化！）
//     Tensor x = torch::randn({2, 50});  // 形状: [batch_size, input_dim]
    
//     // 4. 确保输入在正确设备上
//     auto model_device = network->parameters()[0].device();
//     x = x.to(model_device);
    
//     std::cout << "Input x device: " << x.device() << std::endl;
//     std::cout << "Input shape: " << x.sizes() << std::endl;
    
//     // 5. 前向传播
//     Tensor output = network->forward(x);
    
//     std::cout << "Output device: " << output.device() << std::endl;
//     std::cout << "Output shape: " << output.sizes() << std::endl;
//     std::cout << "Output: " << output << std::endl;
    
//     return 0;
// }