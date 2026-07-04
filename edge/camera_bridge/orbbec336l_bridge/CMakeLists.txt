cmake_minimum_required(VERSION 3.10)
project(visionops_orbbec336l_bridge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

set(ORBBEC_INCLUDE_DIR "" CACHE PATH "Orbbec SDK include directory")
set(ORBBEC_LIB_DIR "" CACHE PATH "Orbbec SDK library directory")
set(ORBBEC_SDK_ROOT "" CACHE PATH "Orbbec SDK root")

if(NOT ORBBEC_INCLUDE_DIR AND ORBBEC_SDK_ROOT)
  set(ORBBEC_INCLUDE_DIR "${ORBBEC_SDK_ROOT}/include")
endif()
if(NOT ORBBEC_LIB_DIR AND ORBBEC_SDK_ROOT)
  set(ORBBEC_LIB_DIR "${ORBBEC_SDK_ROOT}/lib")
endif()

find_library(ORBBEC_SDK_LIBRARY
  NAMES OrbbecSDK libOrbbecSDK.so
  PATHS "${ORBBEC_LIB_DIR}" "${ORBBEC_SDK_ROOT}/lib" "${ORBBEC_SDK_ROOT}/tools"
  NO_DEFAULT_PATH
)

if(NOT ORBBEC_INCLUDE_DIR OR NOT EXISTS "${ORBBEC_INCLUDE_DIR}")
  message(FATAL_ERROR "ORBBEC_INCLUDE_DIR is required")
endif()
if(NOT ORBBEC_SDK_LIBRARY)
  message(FATAL_ERROR "libOrbbecSDK.so was not found. Set ORBBEC_LIB_DIR.")
endif()

add_executable(visionops_orbbec336l_bridge visionops_orbbec336l_bridge.cpp)
target_include_directories(visionops_orbbec336l_bridge PRIVATE "${ORBBEC_INCLUDE_DIR}" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(visionops_orbbec336l_bridge PRIVATE ${ORBBEC_SDK_LIBRARY} ${OpenCV_LIBS} pthread)
