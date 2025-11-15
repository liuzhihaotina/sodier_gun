#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <memory>

// CIFAR-10 数据集类
class CIFAR10 : public torch::data::Dataset<CIFAR10> {
public:
    // 标签映射
    std::vector<std::string> classes = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };

    // 构造函数
    explicit CIFAR10(const std::string& root, bool train = true) {
        std::string data_file;
        if (train) {
            // 训练集有5个批次文件
            for (int i = 1; i <= 5; ++i) {
                data_file = root + "/cifar-10-batches-bin/data_batch_" + std::to_string(i) + ".bin";
                load_batch(data_file);
            }
        } else {
            data_file = root + "/cifar-10-batches-bin/test_batch.bin";
            load_batch(data_file);
        }
    }

    // 获取数据样本
    torch::data::Example<> get(size_t index) override {
        return {images_[index], labels_[index]};
    }

    // 返回数据集大小
    torch::optional<size_t> size() const override {
        return images_.size(0);
    }

    // 获取类别名称
    std::string get_class_name(int64_t label) {
        return classes[label];
    }

private:
    torch::Tensor images_;
    torch::Tensor labels_;

    void load_batch(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("无法打开文件: " + filename);
        }

        const int image_size = 32 * 32 * 3;
        const int batch_size = 10000;
        const int record_size = image_size + 1; // 1字节标签 + 3072字节图像数据

        std::vector<char> buffer(record_size * batch_size);
        file.read(buffer.data(), buffer.size());

        std::vector<float> images;
        std::vector<int64_t> labels;

        for (int i = 0; i < batch_size; ++i) {
            // 读取标签
            int label = static_cast<unsigned char>(buffer[i * record_size]);
            labels.push_back(label);

            // 读取图像数据
            for (int j = 0; j < image_size; ++j) {
                float pixel = static_cast<unsigned char>(buffer[i * record_size + 1 + j]);
                images.push_back(pixel / 255.0f); // 归一化到 [0, 1]
            }
        }

        // 转换为 Tensor
        auto batch_images = torch::from_blob(images.data(), 
            {batch_size, 3, 32, 32}, torch::kFloat).clone();
        auto batch_labels = torch::from_blob(labels.data(), 
            {batch_size}, torch::kInt64).clone();

        // 合并到总数据中
        if (images_.defined()) {
            images_ = torch::cat({images_, batch_images}, 0);
            labels_ = torch::cat({labels_, batch_labels}, 0);
        } else {
            images_ = batch_images;
            labels_ = batch_labels;
        }
    }
};

int main() {
    // 设置随机种子
    torch::manual_seed(1);

    try {
        // 创建数据集
        auto dataset = CIFAR10("../data", true) // true 表示训练集
            .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
            .map(torch::data::transforms::Stack<>());

        // 创建数据加载器
        const size_t batch_size = 4;
        auto data_loader = torch::data::make_data_loader(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(2)
        );

        std::cout << "开始遍历 CIFAR-10 数据..." << std::endl;

        // 获取第一个批次
        auto it = data_loader->begin();
        if (it != data_loader->end()) {
            auto batch = *it;
            auto images = batch.data;
            auto labels = batch.target;

            // 打印形状
            std::cout << "Images shape: " << images.sizes() << std::endl;
            std::cout << "Labels shape: " << labels.sizes() << std::endl;

            // 打印标签对应的类别名称
            std::cout << "Labels: ";
            for (int64_t j = 0; j < labels.size(0); ++j) {
                int64_t label_index = labels[j].item<int64_t>();
                std::cout << ((CIFAR10&)dataset).get_class_name(label_index) << " ";
            }
            std::cout << std::endl;

            // 打印图像统计信息
            std::cout << "Images range: [" << images.min().item<float>() 
                      << ", " << images.max().item<float>() << "]" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        std::cerr << "请确保已经下载 CIFAR-10 数据集到 ./data 目录" << std::endl;
        std::cerr << "可以使用 Python 版本先下载数据:" << std::endl;
        std::cerr << "python -c \"import torchvision; torchvision.datasets.CIFAR10(root='./data', download=True)\"" << std::endl;
    }

    return 0;
}