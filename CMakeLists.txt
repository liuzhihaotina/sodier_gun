cmake_minimum_required(VERSION 3.22.1)

project(LIBTCH)

# 在 CMakeLists.txt 开头设置
set(CMAKE_PREFIX_PATH "/home/tina/miniconda3/envs/myenv")
set(ENV{PATH} "/home/tina/miniconda3/envs/myenv/bin:$ENV{PATH}")
set(ENV{PYTHONPATH} "/home/tina/miniconda3/envs/myenv/lib/python3.9/site-packages")

set(CMAKE_CXX_STANDARD 17)

# 为 Debug 构建自动添加 -g
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -Wall")

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

# 查找 Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# 设置 Python 相关变量
set(PYTHON_EXECUTABLE "/home/tina/miniconda3/envs/myenv/bin/python")
# include_directories(${Python3_INCLUDE_DIRS})
# link_directories(${Python3_LIBRARY_DIRS})

# set(Torch_DIR "/home/tina/libtorch/share/cmake/Torch")
set(Torch_DIR "/home/tina/libtorch_gpu/share/cmake/Torch")
# set(CMAKE_PREFIX_PATH "/home/tina/libtorch_gpu${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

# 明确指定CUDA路径 source ~/.bashrc 已设置，这里可以不用再设置
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6")
# set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")

include_directories(${CMAKE_SOURCE_DIR}/include) # 绝对路径
# include_directories(include) # 相对路径

add_executable(test_libt main.cpp)

target_link_libraries(test_libt "${TORCH_LIBRARIES}")
target_include_directories(test_libt PRIVATE "${TORCH_INCLUDE_DIRS}")



