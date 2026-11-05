# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

add_library(gpu_preprocessor_lp STATIC
  src/folding/folding.cu
  src/folding/folding_kernels.cu
  src/folding/folding_map.cu
  src/folding/folding_reduce.cu
  src/folding/folding_refinement.cu
  src/presolve/transpose64.cu
  src/presolve/device_recovery.cu
  src/presolve/identity_plan.cu
  src/presolve/fixed_col_tape.cu
  src/presolve/gpu_postsolve.cu
  src/presolve/fixed_variable_tape.cu
  src/presolve/postsolve/postsolve_piecewise_linear_reformulation_absolute_value.cu
  src/presolve/postsolve/postsolve_piecewise_linear_reformulation_variable_pair.cu
  src/presolve/postsolve/postsolve_row_sparsification.cu
  src/presolve/gpu_presolve.cu
  src/presolve/rules/rule_row_sparsification.cu
  src/presolve/gpu_presolve_kernels.cu
  src/presolve/presolve_structs.cu
  src/presolve/rules/piecewise_linear_reformulation_variable_pair/shifted_antipodal.cu
  src/presolve/rules/rule_two_row_variable_elimination.cu
  src/presolve/rules/rule_column_domination.cu
  src/presolve/rules/rule_infeasible_and_fixed_variables.cu
  src/presolve/rules/rule_infeasible_and_redundant_rows.cu
  src/presolve/rules/rule_doubleton_equations.cu
  src/presolve/rules/rule_dual_fix.cu
  src/presolve/rules/empty_rows_and_columns/columns.cu
  src/presolve/rules/empty_rows_and_columns/rows.cu
  src/presolve/rules/rule_piecewise_linear_reformulation_variable_pair.cu
  src/presolve/rules/piecewise_linear_reformulation_variable_pair/epigraph.cu
  src/presolve/rules/rule_parallel_columns.cu
  src/presolve/rules/rule_parallel_rows.cu
  src/presolve/rules/rule_zero_cost_product_variables.cu
  src/presolve/rules/rule_bound_strengthening.cu
  src/presolve/rules/rule_redundant_variable_bounds.cu
  src/presolve/rules/rule_singleton_columns.cu
  src/presolve/rules/rule_singleton_rows.cu
  src/presolve/rules/rule_piecewise_linear_reformulation_absolute_value.cu
)
add_library(GPUPreprocessor::lp ALIAS gpu_preprocessor_lp)
target_include_directories(gpu_preprocessor_lp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(gpu_preprocessor_lp PUBLIC cxx_std_17 cuda_std_17)
target_compile_definitions(gpu_preprocessor_lp PUBLIC
  GPUPREPROCESSOR_USE_INT64_NNZ=$<BOOL:${GPUPREPROCESSOR_USE_INT64_NNZ}>)
target_link_libraries(gpu_preprocessor_lp PUBLIC CUDA::cudart CUDA::cusparse CUDA::cublas Threads::Threads)
set_target_properties(gpu_preprocessor_lp PROPERTIES
  EXPORT_NAME lp OUTPUT_NAME gpu_preprocessor_lp POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

install(TARGETS gpu_preprocessor_lp EXPORT GPUPreprocessorTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY examples/ DESTINATION ${CMAKE_INSTALL_DATADIR}/GPUPreprocessor/lp/examples)

if(GPUPREPROCESSOR_BUILD_EXAMPLES)
  add_executable(gpu_preprocessor_example examples/presolve_example.cpp)
  target_link_libraries(gpu_preprocessor_example PRIVATE GPUPreprocessor::lp)
  set_target_properties(gpu_preprocessor_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
  if(GPUPREPROCESSOR_BUILD_TESTS)
    foreach(folding IN ITEMS false true)
      foreach(presolve IN ITEMS false true)
        add_test(NAME example_folding_${folding}_presolve_${presolve}
          COMMAND gpu_preprocessor_example --folding ${folding} --presolve ${presolve})
      endforeach()
    endforeach()
  endif()
endif()

if(GPUPREPROCESSOR_BUILD_TOOLS OR GPUPREPROCESSOR_BUILD_TESTS)
  add_library(gpu_preprocessor_lp_mps_reader STATIC
    mps_reader/src/core/lp_model.cpp
    mps_reader/src/core/sparse_matrix.cpp
    mps_reader/src/io/mpsreader.cpp)
  target_compile_features(gpu_preprocessor_lp_mps_reader PUBLIC cxx_std_17)
  target_include_directories(gpu_preprocessor_lp_mps_reader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/mps_reader/include)
endif()
if(GPUPREPROCESSOR_BUILD_TOOLS)
  add_executable(gpu_preprocessor_postsolve_replay
    tools/postsolve_replay/main.cpp tools/postsolve_replay/snapshot.cpp tools/postsolve_replay/kkt.cu)
  target_link_libraries(gpu_preprocessor_postsolve_replay PRIVATE GPUPreprocessor::lp ${CMAKE_DL_LIBS})
  target_link_options(gpu_preprocessor_postsolve_replay PRIVATE "-Wl,--wrap=cudaMemcpy")
  set_target_properties(gpu_preprocessor_postsolve_replay PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
  install(TARGETS gpu_preprocessor_postsolve_replay RUNTIME DESTINATION bin)
  foreach(tool IN ITEMS gpu_preprocessor_mps gpu_preprocessor_export_reduced gpu_preprocessor_check_postsolve_solution)
    add_executable(${tool} tools/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE GPUPreprocessor::lp gpu_preprocessor_lp_mps_reader)
    set_target_properties(${tool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    install(TARGETS ${tool} RUNTIME DESTINATION bin)
  endforeach()
endif()
if(GPUPREPROCESSOR_BUILD_TESTS)
  add_executable(test_command_line tests/test_command_line.cpp)
  target_include_directories(test_command_line PRIVATE include)
  target_compile_features(test_command_line PRIVATE cxx_std_17)
  target_compile_definitions(test_command_line PRIVATE
    GPUPREPROCESSOR_USE_INT64_NNZ=$<BOOL:${GPUPREPROCESSOR_USE_INT64_NNZ}>)
  target_compile_options(test_command_line PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-UNDEBUG>)
  add_test(NAME test_command_line COMMAND test_command_line)
endif()

if(GPUPREPROCESSOR_BUILD_TESTS)
  add_executable(test_postsolve_kkt tests/test_postsolve_kkt.cpp tools/postsolve_replay/kkt.cu)
  target_link_libraries(test_postsolve_kkt PRIVATE GPUPreprocessor::lp)
  target_compile_options(test_postsolve_kkt PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-UNDEBUG>)
  add_test(NAME test_postsolve_kkt COMMAND test_postsolve_kkt)
  add_executable(test_structural_postsolve_kkt tests/test_structural_postsolve_kkt.cpp tools/postsolve_replay/kkt.cu tools/postsolve_replay/snapshot.cpp)
  target_link_libraries(test_structural_postsolve_kkt PRIVATE GPUPreprocessor::lp)
  target_compile_options(test_structural_postsolve_kkt PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-UNDEBUG>)
  add_test(NAME test_structural_postsolve_kkt COMMAND test_structural_postsolve_kkt)
  add_executable(test_projected_postsolve_kkt tests/test_projected_postsolve_kkt.cpp tools/postsolve_replay/kkt.cu tools/postsolve_replay/snapshot.cpp)
  target_link_libraries(test_projected_postsolve_kkt PRIVATE GPUPreprocessor::lp)
  target_compile_options(test_projected_postsolve_kkt PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-UNDEBUG>)
  add_test(NAME test_projected_postsolve_kkt COMMAND test_projected_postsolve_kkt)
  add_executable(test_shared_bound_recovery tests/test_shared_bound_recovery.cpp tools/postsolve_replay/snapshot.cpp)
  target_link_libraries(test_shared_bound_recovery PRIVATE GPUPreprocessor::lp)
  target_compile_options(test_shared_bound_recovery PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-UNDEBUG>)
  add_test(NAME test_shared_bound_recovery COMMAND test_shared_bound_recovery)
  add_executable(test_postsolve_snapshot tests/test_postsolve_snapshot.cpp tools/postsolve_replay/snapshot.cpp)
  target_link_libraries(test_postsolve_snapshot PRIVATE GPUPreprocessor::lp)
  target_compile_options(test_postsolve_snapshot PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-UNDEBUG>)
  add_test(NAME test_postsolve_snapshot COMMAND test_postsolve_snapshot)
  foreach(test IN ITEMS
    test_compatibility
    test_deferred_schedule
    test_deferred_rules
    test_identity_plan
    test_cleanup_batch
    test_cleanup_trigger
    test_cleanup_workspace
    test_startup_rows
    test_cleanup_ordinary
    test_fixed_variable_tape
    test_device_recovery
    test_structural_device_recovery
    test_folding
    test_model_upload
    test_bound_workspace_oom
    test_packed_bound_recovery
    test_folding_pipeline
    test_gpu_direct_port
    test_gpu_postsolve
    test_postsolve_oom
    test_parallel_fixed_postsolve
    test_parallel_bound_postsolve
    test_parallel_subcol_postsolve
    test_parallel_components_postsolve
    test_gpu_presolve_kernels
    test_rule_piecewise_linear_reformulation_variable_pair_shifted_antipodal
    test_rule_singleton_columns
    test_rule_column_domination
    test_rule_doubleton_equations
    test_rule_dual_fix
    test_rule_parallel_columns
    test_rule_parallel_rows
    test_rule_empty_columns
    test_rule_empty_rows
    test_rule_bound_strengthening
    test_bound_fixing_only
    test_rule_infeasible_and_fixed_variables
    test_rule_infeasible_and_redundant_rows
    test_rule_piecewise_linear_reformulation_variable_pair_epigraph
    test_rule_projected_auxiliaries
    test_rule_redundant_variable_bounds
    test_rule_singleton_rows
    test_rule_piecewise_linear_reformulation_absolute_value
  )
    add_executable(${test} tests/${test}.cpp)
    target_link_libraries(${test} PRIVATE GPUPreprocessor::lp)
    target_compile_options(${test} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-UNDEBUG>)
    add_test(NAME ${test} COMMAND ${test})
  endforeach()
  target_link_options(test_structural_device_recovery PRIVATE
    "-Wl,--wrap=cudaMemcpy" "-Wl,--wrap=cudaMemcpyAsync")
  target_link_options(test_model_upload PRIVATE "-Wl,--wrap=cudaMalloc")
  target_link_options(test_postsolve_oom PRIVATE
    "-Wl,--wrap=cudaMalloc" "-Wl,--wrap=cudaFree" "-Wl,--wrap=cudaFreeAsync" "-Wl,--wrap=cudaMemcpy")
  target_link_options(test_bound_workspace_oom PRIVATE "-Wl,--wrap=cudaMallocAsync")
  target_link_options(test_parallel_bound_postsolve PRIVATE "-Wl,--wrap=cudaMalloc")
  target_link_options(test_parallel_subcol_postsolve PRIVATE "-Wl,--wrap=cudaMalloc")
  target_link_options(test_parallel_components_postsolve PRIVATE "-Wl,--wrap=cudaMalloc")
  target_sources(test_parallel_components_postsolve PRIVATE tests/parallel_components_plan.cu)
  target_compile_options(test_parallel_components_postsolve PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>)
  add_executable(test_mpsreader
    mps_reader/tests/test_main.cpp
    mps_reader/tests/test_mpsreader.cpp)
  target_link_libraries(test_mpsreader PRIVATE gpu_preprocessor_lp_mps_reader)
  add_test(NAME test_mpsreader COMMAND test_mpsreader)
endif()

if(GPUPREPROCESSOR_BUILD_TESTS)
  add_executable(test_doubleton_long_rows tests/test_doubleton_long_rows.cu)
  target_link_libraries(test_doubleton_long_rows PRIVATE GPUPreprocessor::lp)
  add_test(NAME test_doubleton_long_rows COMMAND test_doubleton_long_rows)
endif()

if(GPUPREPROCESSOR_BUILD_TESTS)
  add_executable(test_parallel_signature tests/test_parallel_signature.cu)
  target_link_libraries(test_parallel_signature PRIVATE GPUPreprocessor::lp)
  target_compile_options(test_parallel_signature PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>)
  add_test(NAME test_parallel_signature COMMAND test_parallel_signature)
  add_executable(test_live_load_balance tests/test_live_load_balance.cu)
  target_link_libraries(test_live_load_balance PRIVATE GPUPreprocessor::lp)
  add_test(NAME test_live_load_balance COMMAND test_live_load_balance)
endif()

if(GPUPREPROCESSOR_BUILD_TESTS AND GPUPREPROCESSOR_BUILD_TOOLS)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME test_mps_tool_reporting
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mps_tool_reporting.py
              $<TARGET_FILE:gpu_preprocessor_mps>
              $<TARGET_FILE:gpu_preprocessor_check_postsolve_solution>
              $<TARGET_FILE:gpu_preprocessor_export_reduced>)
  endif()
endif()

if(GPUPREPROCESSOR_BUILD_TESTS)
  add_executable(test_public_pipeline tests/test_public_pipeline.cpp)
  target_link_libraries(test_public_pipeline PRIVATE GPUPreprocessor::lp gpu_preprocessor_lp_mps_reader)
  target_compile_options(test_public_pipeline PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-fno-elide-constructors>)
  add_test(NAME test_public_pipeline COMMAND test_public_pipeline)
endif()
