cmake_minimum_required(VERSION 3.22.1)

project(MAIN1028)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(include) # 相对路径
set(SRC src/test1.cpp src/test2.cpp) # 设置变量名 
add_library(test SHARED ${SRC}) # 添加源文件库

add_executable(main1028 main.cpp)
target_link_libraries(main1028 test) # 链接库





