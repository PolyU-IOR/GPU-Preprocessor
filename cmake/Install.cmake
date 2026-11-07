include(CMakePackageConfigHelpers)
set(install_targets)
if(TARGET gpupresolve_qp)
  list(APPEND install_targets gpupresolve_qp)
  install(FILES qp/include/gpupresolve_qp.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(DIRECTORY qp/examples/ DESTINATION ${CMAKE_INSTALL_DATADIR}/GPUPreprocessor/qp/examples)
endif()
foreach(app presolve presolve_example)
  if(TARGET ${app})
    list(APPEND install_targets ${app})
  endif()
endforeach()
if(install_targets)
  install(TARGETS ${install_targets} EXPORT GPUPreprocessorTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(EXPORT GPUPreprocessorTargets NAMESPACE GPUPreprocessor::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUPreprocessor)
configure_package_config_file(cmake/GPUPreprocessorConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GPUPreprocessorConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUPreprocessor)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/GPUPreprocessorConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GPUPreprocessorConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/GPUPreprocessorConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUPreprocessor)
install(FILES LICENSE NOTICE.md AUTHORS DESTINATION ${CMAKE_INSTALL_DATADIR}/GPUPreprocessor)
# Existing C integrations can keep their package/target names and ABI.
if(GPUPREPROCESSOR_NATIVE_CORE)
  configure_package_config_file(cmake/GPUPreprocessorQPConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/GPUPreprocessorQPConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUPreprocessorQP)
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/GPUPreprocessorQPConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GPUPreprocessorQPConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GPUPreprocessorQPConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GPUPreprocessorQP)
endif()
