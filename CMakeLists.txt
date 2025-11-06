cmake_minimum_required(VERSION 3.22.1)

project(SOLDIERFIRE)

# 在 CMakeLists.txt 开头设置
set(CMAKE_PREFIX_PATH "/home/tina/miniconda3/envs/myenv")
set(ENV{PATH} "/home/tina/miniconda3/envs/myenv/bin:$ENV{PATH}")
set(ENV{PYTHONPATH} "/home/tina/miniconda3/envs/myenv/lib/python3.9/site-packages")

# 查找 Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# 设置 Python 相关变量
set(PYTHON_EXECUTABLE "/home/tina/miniconda3/envs/myenv/bin/python")
# include_directories(${Python3_INCLUDE_DIRS})
# link_directories(${Python3_LIBRARY_DIRS})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


find_package(OpenCV REQUIRED)

# 检查是否找到 OpenCV
if(OpenCV_FOUND)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
message(FATAL_ERROR "OpenCV not found!")
endif()

# include_directories(include) # 相对路径
# 收集指定目录下的所有源文件
# aux_source_directory(src SRC)
# add_library(iter SHARED ${SRC}) # 添加源文件库
set(PYBIND11_INCLUDE_DIR "extern/pybind11/include") # 设置 pybind11 的包含目录
# include_directories(${PYBIND11_INCLUDE_DIR}) # 包含 pybind11 头
include_directories(include ${Python3_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIR}) # 相对路径
# include_directories(${CMAKE_SOURCE_DIR}/include) # 绝对路径
add_subdirectory(extern/pybind11)

# add_executable(soldier_gun_exe bind.cpp src/Gun.cpp src/Soldier.cpp)
# pybind11_add_module(soldier_gun bind.cpp src/Gun.cpp src/Soldier.cpp)
pybind11_add_module(draw src/iter_test.cpp)

# 检查 C++20 支持
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++20" COMPILER_SUPPORTS_CXX20)

if(COMPILER_SUPPORTS_CXX20)
    target_compile_features(draw PRIVATE cxx_std_20)
else()
    target_compile_features(draw PRIVATE cxx_std_17)
    message(WARNING "The compiler ${CMAKE_CXX_COMPILER} has no C++20 support. Using C++17 instead.")
endif()

# 现代 CMake 方式链接库
target_link_libraries(draw PRIVATE ${OpenCV_LIBS})




