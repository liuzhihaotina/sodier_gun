cmake_minimum_required(VERSION 3.22.1)

# 获取年月日，格式为 MMDD
string(TIMESTAMP CURRENT_DATE "%m%d")
# 使用 YYMMDD 格式
# string(TIMESTAMP CURRENT_DATE_WITH_YEAR "%y%m%d")

# 设置项目名称
project(main${CURRENT_DATE})
# message(STATUS "^_^ Project name: main${CURRENT_DATE}")  # --开头
message("^_^ Project name: main${CURRENT_DATE}")    # 无--开头

# 为 Debug 构建自动添加 -g
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

include_directories(include) # 相对路径
# 1 指定源文件
# set(SRC src/test1.cpp src/test2.cpp) # 设置变量名 
# or 2 收集指定目录下的所有源文件
aux_source_directory(src SRC)
add_library(test SHARED ${SRC}) # 添加源文件库

add_executable(${PROJECT_NAME} src/main.cpp)
# 检查 C++20 支持
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++20" COMPILER_SUPPORTS_CXX20)

# -------关于message---------
# message(SEND_ERROR "这是一个错误，但会继续处理")
# message(FATAL_ERROR "这是一个致命错误，CMake 会立即停止")

if(COMPILER_SUPPORTS_CXX23)
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)
    message("^_^ Using C++23.")
elseif(COMPILER_SUPPORTS_CXX20)
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
    message("^_^ Using C++20.")
else()
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
    message(WARNING "The compiler ${CMAKE_CXX_COMPILER} has no C++20 support. Using C++17 instead.")
endif()

# find_package(fmt REQUIRED)

target_link_libraries(${PROJECT_NAME} test)# fmt::fmt) # 链接库
target_include_directories(${PROJECT_NAME} PUBLIC include) # 链接头文件





