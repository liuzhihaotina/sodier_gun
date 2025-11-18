cmake_minimum_required(VERSION 3.22.1)

project(main1118)


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

if(COMPILER_SUPPORTS_CXX20)
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
else()
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
    message(WARNING "The compiler ${CMAKE_CXX_COMPILER} has no C++20 support. Using C++17 instead.")
endif()
target_link_libraries(${PROJECT_NAME} test) # 链接库





