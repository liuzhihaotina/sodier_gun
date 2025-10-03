#ifndef DRAW_UTILS_H
#define DRAW_UTILS_H

/**
 * @file draw_utils.h
 * @brief 绘图工具函数声明
 * 
 * 该头文件声明了用于创建和保存带有几何图形和文字的图像的函数。
 */

/**
 * @brief 创建并保存一个包含填充多边形和文字的图像
 * 
 * 该函数创建一个空白图像，在其中绘制一个填充的绿色多边形，
 * 并在图像上添加"Hello, OpenCV!"文字，最后将图像保存为PNG文件。
 * 
 * @return int 执行状态码
 *             - 0: 成功
 *             - -1: 保存图像失败
 * 
 * @note 输出图像将保存为"output.png"
 * @note 图像尺寸为800x600像素
 * @note 多边形使用绿色填充，文字为黑色
 * 
 * @example
 * @code
 * int result = drw();
 * if (result == 0) {
 *     std::cout << "图像创建成功！" << std::endl;
 * }
 * @endcode
 */
int drw();

#endif // DRAW_UTILS_H