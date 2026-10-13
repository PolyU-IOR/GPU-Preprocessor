cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures")
endif()
project(GPUPreprocessor LANGUAGES CXX CUDA)
find_package(CUDAToolkit 12 REQUIRED)
find_package(Threads REQUIRED)

# Accept the former configuration names when the new name is not specified.
macro(gpu_preprocessor_option name description default_value)
  if(DEFINED GPUPRESOLVER_${name} AND NOT DEFINED GPUPREPROCESSOR_${name})
    set(GPUPREPROCESSOR_${name} "${GPUPRESOLVER_${name}}" CACHE BOOL "${description}")
  endif()
  option(GPUPREPROCESSOR_${name} "${description}" ${default_value})
  set(GPUPRESOLVER_${name} "${GPUPREPROCESSOR_${name}}")
endmacro()
gpu_preprocessor_option(BUILD_TOOLS "Build standalone MPS tools" ${PROJECT_IS_TOP_LEVEL})
gpu_preprocessor_option(BUILD_TESTS "Build numerical regression tests" OFF)
gpu_preprocessor_option(BUILD_EXAMPLES "Build the integration example" ${PROJECT_IS_TOP_LEVEL})
gpu_preprocessor_option(USE_INT64_NNZ "Use int64 nnz and offsets; dimensions stay int32" OFF)
if(GPUPREPROCESSOR_BUILD_TESTS)
  enable_testing()
endif()
add_subdirectory(cpp)
if(GPUPREPROCESSOR_BUILD_EXAMPLES)
  add_executable(gpu_preprocessor_example examples/presolve_example.cpp)
  target_link_libraries(gpu_preprocessor_example PRIVATE GPUPreprocessor::core)
  set_target_properties(gpu_preprocessor_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
  if(GPUPREPROCESSOR_BUILD_TESTS)
    foreach(folding IN ITEMS false true)
      foreach(presolve IN ITEMS false true)
        add_test(NAME example_folding_${folding}_presolve_${presolve}
          COMMAND gpu_preprocessor_example --folding ${folding} --presolve ${presolve})
      endforeach()
    endforeach()
  endif()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS gpu_presolver_core EXPORT GPUPreprocessorTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY cpp/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT GPUPreprocessorTargets NAMESPACE GPUPreprocessor::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUPreprocessor)
configure_package_config_file(cmake/GPUPreprocessorConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GPUPreprocessorConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUPreprocessor)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GPUPreprocessorConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUPreprocessor)
configure_package_config_file(cmake/GPUPresolverCConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GPUPresolverCConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUPresolverC)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GPUPresolverCConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUPresolverC)
install(FILES LICENSE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/GPUPreprocessor)
install(DIRECTORY examples/ DESTINATION ${CMAKE_INSTALL_DATADIR}/GPUPreprocessor/examples)
