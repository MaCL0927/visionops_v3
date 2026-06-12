cmake_minimum_required(VERSION 3.16)

project(visionops_v3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VISIONOPS_BUILD_RUNTIME_MOCK "构建 C++ Runtime Mock HTTP 服务" ON)

if(VISIONOPS_BUILD_RUNTIME_MOCK)
  add_subdirectory(edge/runtime_cpp)
endif()
