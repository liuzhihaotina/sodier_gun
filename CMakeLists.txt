cmake_minimum_required(VERSION 3.22.1)

project(SOLDIERFIRE)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(include) # 相对路径
# include_directories(${CMAKE_SOURCE_DIR}/include) # 绝对路径
add_subdirectory(extern/pybind11)

find_package(OpenCV REQUIRED)

# 检查是否找到 OpenCV
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()



# add_executable(soldier_gun_exe bind.cpp src/Gun.cpp src/Soldier.cpp)
# pybind11_add_module(soldier_gun bind.cpp src/Gun.cpp src/Soldier.cpp)
pybind11_add_module(draw src/draw.cpp)

# 现代 CMake 方式链接库
target_link_libraries(draw PRIVATE ${OpenCV_LIBS})




