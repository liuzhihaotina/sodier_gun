#pragma once
#include <torch/torch.h>

class MyDataset : public torch::data::Dataset<MyDataset> {
public:
    // 构造函数
    MyDataset(const std::string& data_file) {
        // 加载数据
        int num_samples = 10; // 示例样本数量
        data = torch::rand({num_samples, 32, 32, 5});  // 示例数据
        targets = torch::randint(0, 6, {num_samples}); // 示例标签
    }
    
    // 获取单个样本
    torch::data::Example<> get(size_t index) override {
        return {data[index], targets[index]};
    }
    
    // 返回数据集大小
    torch::optional<size_t> size() const override {
        return data.size(0);
    }
    
private:
    torch::Tensor data;
    torch::Tensor targets;
};