set(VISIONOPS_RUNTIME_SOURCES
  src/main.cpp
  src/app_config.cpp
  src/cli_args.cpp
  src/http_server.cpp
  src/json_utils.cpp
  src/model_config.cpp
  src/model_package.cpp
  src/postprocess.cpp
  src/postprocess_detect.cpp
  src/postprocess_classification.cpp
  src/postprocess_obb.cpp
  src/postprocess_seg.cpp
  src/preprocess.cpp
  src/rga_preprocess.cpp
  src/rknn_runner_mock.cpp
  src/rknn_runner_factory.cpp
  src/roi_filter.cpp
  src/runtime_app.cpp
  src/runtime_state.cpp
  src/snapshot_provider.cpp
  src/stream_worker_mock.cpp
)

if(VISIONOPS_ENABLE_RKNN)
  if(NOT VISIONOPS_RKNN_INCLUDE_DIR OR
     NOT EXISTS "${VISIONOPS_RKNN_INCLUDE_DIR}/rknn_api.h")
    message(FATAL_ERROR
      "VISIONOPS_ENABLE_RKNN=ON 时必须设置 VISIONOPS_RKNN_INCLUDE_DIR，且目录中包含 rknn_api.h")
  endif()
  if(NOT VISIONOPS_RKNN_LIBRARY OR NOT EXISTS "${VISIONOPS_RKNN_LIBRARY}")
    message(FATAL_ERROR
      "VISIONOPS_ENABLE_RKNN=ON 时必须设置有效的 VISIONOPS_RKNN_LIBRARY")
  endif()
  list(APPEND VISIONOPS_RUNTIME_SOURCES src/rknn_runner_real.cpp)
endif()

if(VISIONOPS_ENABLE_OPENCV)
  find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
endif()

if(VISIONOPS_ENABLE_RGA)
  if(NOT VISIONOPS_ENABLE_OPENCV)
    message(FATAL_ERROR "VISIONOPS_ENABLE_RGA=ON 需要同时启用 VISIONOPS_ENABLE_OPENCV=ON")
  endif()
  if(NOT VISIONOPS_RGA_INCLUDE_DIR OR
     NOT EXISTS "${VISIONOPS_RGA_INCLUDE_DIR}/rga/im2d.hpp")
    message(FATAL_ERROR
      "VISIONOPS_ENABLE_RGA=ON 时必须设置 VISIONOPS_RGA_INCLUDE_DIR，且目录中包含 rga/im2d.hpp")
  endif()
  if(NOT VISIONOPS_RGA_LIBRARY OR NOT EXISTS "${VISIONOPS_RGA_LIBRARY}")
    message(FATAL_ERROR
      "VISIONOPS_ENABLE_RGA=ON 时必须设置有效的 VISIONOPS_RGA_LIBRARY")
  endif()
endif()

add_executable(visionops_runtime_mock ${VISIONOPS_RUNTIME_SOURCES})

target_include_directories(visionops_runtime_mock
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(VISIONOPS_ENABLE_RKNN)
  target_include_directories(visionops_runtime_mock PRIVATE ${VISIONOPS_RKNN_INCLUDE_DIR})
  target_compile_definitions(visionops_runtime_mock PRIVATE VISIONOPS_HAS_RKNN=1)
  target_link_libraries(visionops_runtime_mock PRIVATE ${VISIONOPS_RKNN_LIBRARY})
endif()

if(VISIONOPS_ENABLE_OPENCV)
  target_compile_definitions(visionops_runtime_mock PRIVATE VISIONOPS_HAS_OPENCV=1)
  target_include_directories(visionops_runtime_mock PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(visionops_runtime_mock PRIVATE ${OpenCV_LIBS})
endif()

if(VISIONOPS_ENABLE_RGA)
  target_compile_definitions(visionops_runtime_mock PRIVATE VISIONOPS_HAS_RGA=1)
  target_include_directories(visionops_runtime_mock PRIVATE ${VISIONOPS_RGA_INCLUDE_DIR})
  target_link_libraries(visionops_runtime_mock PRIVATE ${VISIONOPS_RGA_LIBRARY} ${CMAKE_DL_LIBS})
endif()

target_compile_options(visionops_runtime_mock
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

find_package(Threads REQUIRED)
target_link_libraries(visionops_runtime_mock PRIVATE Threads::Threads)

add_executable(visionops_postprocess_fixture
  tests/postprocess_fixture.cpp
  src/json_utils.cpp
  src/postprocess.cpp
  src/postprocess_detect.cpp
  src/postprocess_classification.cpp
  src/postprocess_obb.cpp
  src/postprocess_seg.cpp
  src/roi_filter.cpp
)
target_include_directories(visionops_postprocess_fixture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(visionops_postprocess_fixture PRIVATE -Wall -Wextra -Wpedantic)
