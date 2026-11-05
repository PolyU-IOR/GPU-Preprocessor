# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()
project(GPUPreprocessor LANGUAGES CXX CUDA)
find_package(CUDAToolkit 12 REQUIRED)
find_package(Threads REQUIRED)

option(GPUPREPROCESSOR_BUILD_TOOLS "Build standalone MPS tools" ${PROJECT_IS_TOP_LEVEL})
option(GPUPREPROCESSOR_BUILD_TESTS "Build numerical regression tests" OFF)
option(GPUPREPROCESSOR_BUILD_EXAMPLES "Build the integration example" ${PROJECT_IS_TOP_LEVEL})
option(GPUPREPROCESSOR_USE_INT64_NNZ "Use int64 nnz and offsets; dimensions stay int32" OFF)
if(GPUPREPROCESSOR_BUILD_TESTS)
  enable_testing()
endif()
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
add_subdirectory(lp)

install(EXPORT GPUPreprocessorTargets NAMESPACE GPUPreprocessor::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUPreprocessor)
configure_package_config_file(cmake/GPUPreprocessorConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GPUPreprocessorConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUPreprocessor)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GPUPreprocessorConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUPreprocessor)
install(FILES LICENSE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/GPUPreprocessor)
