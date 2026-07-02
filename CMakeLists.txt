cmake_minimum_required(VERSION 3.16)

project(visionops_v3 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VISIONOPS_BUILD_RUNTIME_MOCK "构建 C++ Runtime Mock HTTP 服务" ON)
option(VISIONOPS_ENABLE_RKNN "启用真实 RKNN Runtime Runner" OFF)
option(VISIONOPS_ENABLE_OPENCV "启用 OpenCV 测试图片解码（M9.3 默认未使用）" OFF)
option(VISIONOPS_ENABLE_RGA "启用 Rockchip RGA 预处理加速" OFF)
set(VISIONOPS_RKNN_INCLUDE_DIR "" CACHE PATH "RKNN Runtime 头文件目录")
set(VISIONOPS_RKNN_LIBRARY "" CACHE FILEPATH "RKNN Runtime 动态或静态库路径")
set(VISIONOPS_RGA_INCLUDE_DIR "" CACHE PATH "RGA 头文件目录，例如 /usr/include")
set(VISIONOPS_RGA_LIBRARY "" CACHE FILEPATH "RGA 动态库路径，例如 /usr/lib/librga.so")

if(VISIONOPS_BUILD_RUNTIME_MOCK)
  add_subdirectory(edge/runtime_cpp)
endif()
