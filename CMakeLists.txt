cmake_minimum_required(VERSION 3.20)

# Apple Silicon only, macOS 14+ (see docs/DESIGN.md §6).
if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS deployment target")
endif()

project(onnxruntime_mlx_ep LANGUAGES C CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "The MLX EP only builds on macOS (Apple Silicon).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindONNXRuntime)

option(ORT_MLX_BUILD_E2E "Build the standalone end-to-end test" ON)
option(ORT_MLX_BUILD_TESTS "Build the MLX op-correctness tests" ON)

# --- MLX (mlx-c) is now a HARD dependency ------------------------------------------
# The EP translates the whole fused decoder subgraph into an MLX graph; there is no
# hand-kernel fallback. Configure FAILS if mlx-c / mlx are not installed
# (`brew install mlx-c`). Located unconditionally via Homebrew.
execute_process(COMMAND brew --prefix mlx-c
  OUTPUT_VARIABLE MLXC_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE MLXC_BREW_RESULT)
execute_process(COMMAND brew --prefix mlx
  OUTPUT_VARIABLE MLX_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE MLX_BREW_RESULT)
if(NOT MLXC_PREFIX OR NOT MLX_PREFIX)
  message(FATAL_ERROR
    "mlx-c / mlx not found. MLX is a hard dependency of the MLX EP. "
    "Install it with: brew install mlx-c")
endif()
find_library(MLXC_LIB mlxc PATHS ${MLXC_PREFIX}/lib REQUIRED)
find_library(MLX_LIB  mlx  PATHS ${MLX_PREFIX}/lib  REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
message(STATUS "MLX EP: mlx-c=${MLXC_PREFIX} mlx=${MLX_PREFIX}")

# --- The plugin EP shared library -------------------------------------------------
add_library(onnxruntime_mlx_ep SHARED
  src/ep/entry.cc
  src/ep/ep_factory.cc
  src/ep/ep.cc
  src/ep/mlx_backend.cc
  src/ep/op_registry.cc
  src/ep/ops/elementwise.cc
  src/ep/ops/norm.cc
  src/ep/ops/attention.cc
  src/ep/ops/quant.cc
  src/ep/data_transfer.cc
  src/ep/metal_context.mm)

target_include_directories(onnxruntime_mlx_ep PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ep
  ${MLXC_PREFIX}/include
  ${ORT_INCLUDE_DIR})

# Hide everything except the two exported entry points (which use visibility("default")).
target_compile_options(onnxruntime_mlx_ep PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden -Wall)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

target_link_libraries(onnxruntime_mlx_ep PRIVATE
  onnxruntime::onnxruntime
  ${MLXC_LIB}
  ${MLX_LIB}
  ${ACCELERATE_FRAMEWORK}
  ${METAL_FRAMEWORK}
  ${FOUNDATION_FRAMEWORK}
  ${QUARTZCORE_FRAMEWORK})

set_target_properties(onnxruntime_mlx_ep PROPERTIES
  OUTPUT_NAME onnxruntime_mlx_ep
  BUILD_RPATH "${ORT_HOME}/lib;${MLXC_PREFIX}/lib;${MLX_PREFIX}/lib"
  INSTALL_RPATH "${ORT_HOME}/lib;${MLXC_PREFIX}/lib;${MLX_PREFIX}/lib")

# --- Standalone end-to-end test ---------------------------------------------------
if(ORT_MLX_BUILD_E2E)
  add_executable(mlx_e2e tests/e2e/e2e_test.cc)
  target_include_directories(mlx_e2e PRIVATE ${ORT_INCLUDE_DIR})
  target_link_libraries(mlx_e2e PRIVATE onnxruntime::onnxruntime)
  set_target_properties(mlx_e2e PROPERTIES
    BUILD_RPATH "${ORT_HOME}/lib"
    INSTALL_RPATH "${ORT_HOME}/lib")
endif()

# --- MLX op-correctness tests (each translated decoder op via MLX vs ORT CPU) ------
if(ORT_MLX_BUILD_TESTS)
  enable_testing()
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    get_filename_component(ORT_LIBRARY_DIR "${ORT_LIBRARY}" DIRECTORY)
    add_test(NAME mlx_op_tests
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/ops/mlx_op_test.py
        $<TARGET_FILE:onnxruntime_mlx_ep>)
    set_tests_properties(mlx_op_tests PROPERTIES
      ENVIRONMENT "DYLD_LIBRARY_PATH=${ORT_LIBRARY_DIR}")
  endif()
endif()

# --- MLX EP session-lifetime leak regression --------------------------------------
if(ORT_MLX_BUILD_E2E)
  enable_testing()

  add_executable(mlx_leak_test tests/e2e/leak_test.mm)
  target_include_directories(mlx_leak_test PRIVATE ${ORT_INCLUDE_DIR})
  target_compile_options(mlx_leak_test PRIVATE -Wall -fno-objc-arc)
  target_link_libraries(mlx_leak_test PRIVATE
    onnxruntime::onnxruntime
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK})
  set_target_properties(mlx_leak_test PROPERTIES
    BUILD_RPATH "${ORT_HOME}/lib"
    INSTALL_RPATH "${ORT_HOME}/lib")

  set(ORT_MLX_E2E_MODEL
    "${CMAKE_CURRENT_SOURCE_DIR}/../onnx-genai/models/qwen2.5-0.5b-cpu-recipe/model.onnx"
    CACHE FILEPATH "Qwen model used by MLX EP E2E and leak tests")
  get_filename_component(ORT_MLX_ORT_LIBRARY_DIR "${ORT_LIBRARY}" DIRECTORY)
  add_test(NAME mlx_leak_test
    COMMAND $<TARGET_FILE:mlx_leak_test>
      "${ORT_MLX_E2E_MODEL}"
      $<TARGET_FILE:onnxruntime_mlx_ep>
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/prompt_tokens.txt"
      8
      2)
  set_tests_properties(mlx_leak_test PROPERTIES
    ENVIRONMENT "DYLD_LIBRARY_PATH=${ORT_MLX_ORT_LIBRARY_DIR}"
    SKIP_RETURN_CODE 77
    TIMEOUT 600)

  # E2E coherence gate as a ctest: full-MLX prefill+decode must stream coherently.
  if(ORT_MLX_BUILD_E2E)
    add_test(NAME mlx_e2e
      COMMAND $<TARGET_FILE:mlx_e2e>
        "${ORT_MLX_E2E_MODEL}"
        $<TARGET_FILE:onnxruntime_mlx_ep>
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/prompt_tokens.txt"
        8)
    set_tests_properties(mlx_e2e PROPERTIES
      ENVIRONMENT "DYLD_LIBRARY_PATH=${ORT_MLX_ORT_LIBRARY_DIR}"
      SKIP_RETURN_CODE 77
      TIMEOUT 600)
  endif()
endif()
