add_executable(visionops_runtime_mock
  src/main.cpp
  src/http_server.cpp
  src/runtime_state.cpp
  src/mock_data.cpp
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
