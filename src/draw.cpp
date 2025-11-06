// #include <pybind11/pybind11.h>
// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <chrono>
// #include "draw.h"

// int drw() {
//     // 创建一个空白图像
//     cv::Mat image = cv::Mat::zeros(600, 800, CV_8UC3);

//     // 定义多边形的顶点
//     std::vector<cv::Point> polygonPoints = {
//         cv::Point(400, 100),
//         cv::Point(100, 300),
//         cv::Point(200, 500),
//         cv::Point(600, 500),
//         cv::Point(700, 300)
//     };

//     // 填充多边形
//     cv::fillPoly(image, std::vector<std::vector<cv::Point>>{polygonPoints}, cv::Scalar(0, 255, 0));

//     // 设置字体属性
//     int fontFace = cv::FONT_HERSHEY_SIMPLEX;
//     double fontScale = 1;
//     int thickness = 2;

//     // 写入文字
//     std::string text = "Hello, OpenCV!";
//     cv::putText(image, text, cv::Point(300, 250), fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);

//     // 保存图片文件
//     if (!cv::imwrite("output.png", image)) {
//         std::cerr << "Failed to save image" << std::endl;
//         return -1;
//     }

//     std::cout << "Image saved as output.png" << std::endl;

//     return 0;
// }

// void caul(){
//     std::cout<<"cpp--开始计算耗时函数"<<std::endl;
//     auto start = std::chrono::high_resolution_clock::now();
//     float sum = 0.0;
//     for (int i = 0; i < 2000000000; ++i) {
//         sum += std::sqrt(i); 
//     }
//     std::cout<< "Sum: " << sum << std::endl;
//     // 结束时间
//     auto end = std::chrono::high_resolution_clock::now();
    
//     // 计算持续时间
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
//     std::cout << "耗时: " << duration.count()/1e+6 << " 秒" << std::endl;
// }

// PYBIND11_MODULE(draw, m) { //(A,m) 的A要与cmake的pybind11_add_module(A src/xxx.cpp)的A 一致
//     m.doc() = "plot";
//     m.def("drw", &drw, "A function which draws with opencv");
//     m.def("caul", &caul, "计算耗时函数");
// }