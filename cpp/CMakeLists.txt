add_library(gpu_presolver_core STATIC
  src/model/lp_problem.cpp
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
  src/presolve/gpu_presolve.cu
  src/presolve/gpu_presolve_kernels.cu
  src/presolve/presolve_structs.cu
  src/presolve/rules/variable_pair_aggregation/shifted_antipodal.cu
  src/presolve/rules/rule_two_row_variable_elimination.cu
  src/presolve/rules/rule_column_domination.cu
  src/presolve/rules/rule_infeasible_and_fixed_variables.cu
  src/presolve/rules/rule_infeasible_and_redundant_rows.cu
  src/presolve/rules/rule_doubleton_equations.cu
  src/presolve/rules/rule_dual_fix.cu
  src/presolve/rules/empty_rows_and_columns/columns.cu
  src/presolve/rules/empty_rows_and_columns/rows.cu
  src/presolve/rules/rule_variable_pair_aggregation.cu
  src/presolve/rules/variable_pair_aggregation/epigraph.cu
  src/presolve/rules/rule_parallel_columns.cu
  src/presolve/rules/rule_parallel_rows.cu
  src/presolve/rules/rule_mccormick_variable_elimination.cu
  src/presolve/rules/rule_bound_strengthening.cu
  src/presolve/rules/rule_redundant_variable_bounds.cu
  src/presolve/rules/rule_singleton_columns.cu
  src/presolve/rules/rule_singleton_rows.cu
  src/presolve/rules/rule_absolute_value_epigraph_reduction.cu
)
add_library(GPUPreprocessor::core ALIAS gpu_presolver_core)
add_library(GPUPresolverC::core ALIAS gpu_presolver_core)
target_include_directories(gpu_presolver_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(gpu_presolver_core PUBLIC cxx_std_17 cuda_std_17)
target_compile_definitions(gpu_presolver_core PUBLIC
  GPUPRESOLVER_USE_INT64_NNZ=$<BOOL:${GPUPRESOLVER_USE_INT64_NNZ}>)
target_link_libraries(gpu_presolver_core PUBLIC CUDA::cudart CUDA::cusparse Threads::Threads)
set_target_properties(gpu_presolver_core PROPERTIES
  EXPORT_NAME core OUTPUT_NAME gpu_preprocessor POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(GPUPRESOLVER_BUILD_TOOLS OR GPUPRESOLVER_BUILD_TESTS)
  add_library(gpu_presolver_mps_reader STATIC
    mps_reader/src/core/lp_model.cpp
    mps_reader/src/core/sparse_matrix.cpp
    mps_reader/src/io/mpsreader.cpp)
  target_compile_features(gpu_presolver_mps_reader PUBLIC cxx_std_17)
  target_include_directories(gpu_presolver_mps_reader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/mps_reader/include)
endif()
if(GPUPRESOLVER_BUILD_TOOLS)
  foreach(tool IN ITEMS gpu_preprocessor_mps gpu_preprocessor_export_reduced gpu_preprocessor_check_postsolve_solution)
    add_executable(${tool} tools/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE GPUPresolverC::core gpu_presolver_mps_reader)
    set_target_properties(${tool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    install(TARGETS ${tool} RUNTIME DESTINATION bin)
  endforeach()
endif()
if(GPUPRESOLVER_BUILD_TESTS)
  add_executable(test_command_line tests/test_command_line.cpp)
  target_include_directories(test_command_line PRIVATE include)
  target_compile_features(test_command_line PRIVATE cxx_std_17)
  target_compile_definitions(test_command_line PRIVATE
    GPUPRESOLVER_USE_INT64_NNZ=$<BOOL:${GPUPRESOLVER_USE_INT64_NNZ}>)
  target_compile_options(test_command_line PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-UNDEBUG>)
  add_test(NAME test_command_line COMMAND test_command_line)
  foreach(test IN ITEMS
    test_deferred_schedule
    test_deferred_rules
    test_identity_plan
    test_cleanup_batch
    test_cleanup_full
    test_cleanup_closure
    test_cleanup_trigger
    test_cleanup_workspace
    test_startup_rows
    test_cleanup_ordinary
    test_fixed_col_tape
    test_device_recovery
    test_structural_device_recovery
    test_folding
    test_folding_pipeline
    test_gpu_direct_port
    test_gpu_postsolve
    test_gpu_presolve_kernels
    test_lp_problem
    test_rule_variable_pair_shifted_antipodal
    test_rule_singleton_columns
    test_rule_column_domination
    test_rule_doubleton_equations
    test_rule_dual_fix
    test_rule_parallel_columns
    test_rule_parallel_rows
    test_rule_empty_columns
    test_rule_empty_rows
    test_rule_bound_strengthening
    test_rule_infeasible_and_fixed_variables
    test_rule_infeasible_and_redundant_rows
    test_rule_variable_pair_epigraph
    test_rule_projected_auxiliaries
    test_rule_redundant_variable_bounds
    test_rule_singleton_rows
    test_rule_absolute_value_epigraph_reduction
  )
    add_executable(${test} tests/${test}.cpp)
    target_link_libraries(${test} PRIVATE GPUPresolverC::core)
    target_compile_options(${test} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-UNDEBUG>)
    add_test(NAME ${test} COMMAND ${test})
  endforeach()
  if(UNIX AND NOT APPLE)
    target_link_options(test_structural_device_recovery PRIVATE
      "-Wl,--wrap=cudaMemcpy" "-Wl,--wrap=cudaMemcpyAsync")
  endif()
  add_executable(test_mpsreader
    mps_reader/tests/test_main.cpp
    mps_reader/tests/test_mpsreader.cpp)
  target_link_libraries(test_mpsreader PRIVATE gpu_presolver_mps_reader)
  add_test(NAME test_mpsreader COMMAND test_mpsreader)
endif()

if(GPUPRESOLVER_BUILD_TESTS)
  add_executable(test_doubleton_long_rows tests/test_doubleton_long_rows.cu)
  target_link_libraries(test_doubleton_long_rows PRIVATE GPUPresolverC::core)
  add_test(NAME test_doubleton_long_rows COMMAND test_doubleton_long_rows)
endif()

if(GPUPRESOLVER_BUILD_TESTS)
  add_executable(test_parallel_signature tests/test_parallel_signature.cu)
  target_link_libraries(test_parallel_signature PRIVATE GPUPresolverC::core)
  target_compile_options(test_parallel_signature PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>)
  add_test(NAME test_parallel_signature COMMAND test_parallel_signature)
  add_executable(test_live_load_balance tests/test_live_load_balance.cu)
  target_link_libraries(test_live_load_balance PRIVATE GPUPresolverC::core)
  add_test(NAME test_live_load_balance COMMAND test_live_load_balance)
endif()

if(GPUPRESOLVER_BUILD_TESTS)
  add_executable(test_public_pipeline tests/test_public_pipeline.cpp)
  target_link_libraries(test_public_pipeline PRIVATE GPUPreprocessor::core gpu_presolver_mps_reader)
  target_compile_options(test_public_pipeline PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-fno-elide-constructors>)
  add_test(NAME test_public_pipeline COMMAND test_public_pipeline)
endif()
