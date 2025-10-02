cmake_minimum_required(VERSION 3.22.1)

project(SOLDIERFIRE)

include_directories(${CMAKE_SOURCE_DIR}/include) # 绝对路径
# include_directories(include) # 相对路径

add_executable(my_cmake_exe main.cpp src/Gun.cpp src/Soldier.cpp)
