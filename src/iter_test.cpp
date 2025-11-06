#include <pybind11/pybind11.h>
#include <iostream>
#include <memory>
#include <vector>
#include <list>
#include <stdexcept>
#include <functional>
#include <chrono>
#include <thread>
#include <ranges>

#include "iter.h"

// 示例使用
int main() {
    auto numbers_view = std::views::iota(1, 100000001);  // 生成 1-1000 的视图
    
    // 转换为 vector
    std::vector<int> numbers(numbers_view.begin(), numbers_view.end());
    // 示例1：基本使用
    // std::vector<int> numbers = {1, 2, 3, 4, 5};
    // auto start = std::chrono::high_resolution_clock::now();
    auto it = advanced_iter(numbers);
    
    try {
        while (true) {
            int a=it.next();
            if (a % 10000000 == 0){
                std::cout << "已迭代到: " << a << ' ';// << std::endl;
            }
            // std::cout << a << " ";
        }
    } catch (const StopIteration&) {
        std::cout << "\n迭代结束" << std::endl;
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "执行时间: " << duration.count() / 1e+6 << " 秒" << std::endl;

    // // 示例2：使用默认值
    // auto it2 = iter(numbers);
    // std::cout << next(it2) << std::endl;  // 1
    // std::cout << next(it2) << std::endl;  // 2
    // std::cout << next(it2, -1) << std::endl;  // 3
    // // ... 迭代完所有元素后
    // std::cout << next(it2, -1) << std::endl;  // -1 (默认值)

    // // 示例3：字符串迭代
    // std::vector<std::string> words = {"hello", "world", "cpp"};
    // auto word_it = advanced_iter(words);
    
    // try {
    //     std::cout << word_it.next() << std::endl;  // hello
    //     std::cout << word_it.next() << std::endl;  // world
    //     std::cout << word_it.next() << std::endl;  // cpp
    //     std::cout << word_it.next() << std::endl;  // 抛出 StopIteration
    // } catch (const StopIteration& e) {
    //     std::cout << "捕获异常: " << e.what() << std::endl;
    // }

    return 0;
}
PYBIND11_MODULE(draw, m) { //(A,m) 的A要与cmake的pybind11_add_module(A src/xxx.cpp)的A 一致
    m.doc() = "iter_test module";
    m.def("iter_main", &main, "main function of iter_test");
}