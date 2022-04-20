
# Try to enable code coverage if all dependencies are satisfied
# otherwise, code coverage is disabled and a warning is issued
function(maybe_enable_coverage)
  set(COVERARE_ENABLED OFF PARENT_SCOPE)
  if (NOT ENABLE_COVERAGE)
    return()
  endif()
  # Coverage are only enabled using GNU G++ / GCC and in debug build
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Check if all required dependencies are satisfied
    find_program(GCOVR gcovr)
    find_program(GCOV gcov)
    if (GCOVR AND GCOV)
      # Setup code coverage
      include(extern/CodeCoverage.cmake)
      set(COVERAGE_EXCLUDES "/extern" "/test")
      message(STATUS "Code coverage enabled")
      set(COVERARE_ENABLED ON PARENT_SCOPE)
    else ()
      # Issue a warning if an error occured
      message(WARNING "Cannot enable code coverage tests: gcovr or gcov not found")
    endif ()
  else ()
    message(STATUS "Code coverage unavailable: to enable, ensure the build type is set to DEBUG and the compiler is GNU G++ / GCC")
  endif ()

endfunction()