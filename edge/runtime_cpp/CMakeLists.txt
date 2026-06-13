add_executable(visionops_runtime_mock
  src/main.cpp
  src/app_config.cpp
  src/cli_args.cpp
  src/http_server.cpp
  src/json_utils.cpp
  src/postprocess_detect.cpp
  src/postprocess_obb.cpp
  src/postprocess_seg.cpp
  src/preprocess.cpp
  src/rknn_runner_mock.cpp
  src/runtime_app.cpp
  src/runtime_state.cpp
  src/snapshot_provider.cpp
  src/stream_worker_mock.cpp
)

target_include_directories(visionops_runtime_mock
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(visionops_runtime_mock
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

find_package(Threads REQUIRED)
target_link_libraries(visionops_runtime_mock PRIVATE Threads::Threads)
