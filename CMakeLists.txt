cmake_minimum_required(VERSION 3.20)

# Apple Silicon only, macOS 14+ (see docs/DESIGN.md 6).
if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS deployment target")
endif()

project(onnxruntime_mps_ep LANGUAGES C CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "The Metal/MPS EP only builds on macOS (Apple Silicon).")
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

option(ORT_MPS_BUILD_E2E "Build the standalone end-to-end test" ON)
option(ORT_MPS_BUILD_TESTS "Build Metal kernel correctness tests" ON)

# Runtime compilation is retained because the standalone `metallib` tool is not guaranteed to
# be installed. Generate one C++ raw string directly from the canonical .metal files so the
# runtime source cannot drift from the reviewable shaders.
set(ORT_MPS_METAL_KERNEL_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/elementwise.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/rope.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/gather_block_quantized.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/data_movement.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/matmulnbits.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/norm.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/softmax.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/gqa.metal)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ORT_MPS_METAL_KERNEL_FILES})
set(METAL_KERNEL_SOURCE "")
foreach(kernel_file IN LISTS ORT_MPS_METAL_KERNEL_FILES)
  file(READ "${kernel_file}" kernel_source)
  string(APPEND METAL_KERNEL_SOURCE "\n${kernel_source}\n")
endforeach()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file(cmake/metal_kernels.inc.in
  "${CMAKE_CURRENT_BINARY_DIR}/generated/metal_kernels.inc" @ONLY)

# --- The plugin EP shared library -------------------------------------------------
add_library(onnxruntime_mps_ep SHARED
  src/ep/entry.cc
  src/ep/ep_factory.cc
  src/ep/ep.cc
  src/ep/data_transfer.cc
  src/ep/metal_context.mm)

target_include_directories(onnxruntime_mps_ep PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ep
  ${CMAKE_CURRENT_BINARY_DIR}/generated
  ${ORT_INCLUDE_DIR})

# Hide everything except the two exported entry points (which use visibility("default")).
target_compile_options(onnxruntime_mps_ep PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden -Wall)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

target_link_libraries(onnxruntime_mps_ep PRIVATE
  onnxruntime::onnxruntime
  ${METAL_FRAMEWORK}
  ${FOUNDATION_FRAMEWORK}
  ${QUARTZCORE_FRAMEWORK})

set_target_properties(onnxruntime_mps_ep PROPERTIES
  OUTPUT_NAME onnxruntime_mps_ep
  BUILD_RPATH "${ORT_HOME}/lib"
  INSTALL_RPATH "${ORT_HOME}/lib")

# --- Standalone end-to-end test ---------------------------------------------------
if(ORT_MPS_BUILD_E2E)
  add_executable(mps_e2e tests/e2e/e2e_test.cc)
  target_include_directories(mps_e2e PRIVATE ${ORT_INCLUDE_DIR})
  target_link_libraries(mps_e2e PRIVATE onnxruntime::onnxruntime)
  set_target_properties(mps_e2e PROPERTIES
    BUILD_RPATH "${ORT_HOME}/lib"
    INSTALL_RPATH "${ORT_HOME}/lib")
endif()

if(ORT_MPS_BUILD_TESTS)
  enable_testing()
  add_executable(mps_kernel_tests
    tests/kernels/kernel_test.cc
    src/ep/metal_context.mm)
  target_include_directories(mps_kernel_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ep
    ${CMAKE_CURRENT_BINARY_DIR}/generated)
  target_compile_options(mps_kernel_tests PRIVATE -Wall)
  target_link_libraries(mps_kernel_tests PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK})
  add_test(NAME mps_kernel_tests COMMAND mps_kernel_tests)

  # Mariette's hot-path kernel correctness tests (MatMulNBits, RMSNorm,
  # SkipSimplifiedLayerNorm, Softmax) vs CPU reference. Separate target so it
  # merges without touching the shared mps_kernel_tests target.
  add_executable(mps_mariette_tests
    tests/kernels/mariette_kernel_test.cc
    src/ep/metal_context.mm)
  target_include_directories(mps_mariette_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ep
    ${CMAKE_CURRENT_BINARY_DIR}/generated)
  target_compile_options(mps_mariette_tests PRIVATE -Wall)
  target_link_libraries(mps_mariette_tests PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK})
  add_test(NAME mps_mariette_tests COMMAND mps_mariette_tests)

  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    get_filename_component(ORT_LIBRARY_DIR "${ORT_LIBRARY}" DIRECTORY)
    add_test(NAME mps_ort_reference_tests
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/ort_reference_test.py
        $<TARGET_FILE:onnxruntime_mps_ep>)
    set_tests_properties(mps_ort_reference_tests PROPERTIES
      ENVIRONMENT "DYLD_LIBRARY_PATH=${ORT_LIBRARY_DIR}")
  endif()
endif()
