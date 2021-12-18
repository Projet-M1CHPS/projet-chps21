cmake_minimum_required(VERSION 3.16)

project(projet-chps21 VERSION 0.3.0.4 DESCRIPTION "Cancer detection using deep learning" LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)

# Compiler specific flags
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -fast -O3 -qmkl -lm")
  # Intel compilers are not supported for coverage tests
  set(CODE_COVERAGE OFF)
  message(STATUS ${CMAKE_CXX_FLAGS})
else ()
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

  if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -march=native")
  endif ()
endif ()

# Code coverage flags
# Added as an external plugin
include(extern/cmake_coverage.cmake)

# Defines version specific Macro
configure_file(include/tscl/ProjectConstant.hpp.in include/ProjectVersion.hpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

find_package(Threads REQUIRED)
find_package(BLAS REQUIRED)

# We remove googletest and the std from the coverage report
# We also remove the test directory, since templated functions that are used
# will appear in their respective headers
add_code_coverage_all_targets(EXCLUDE ${CMAKE_SOURCE_DIR}/test/* */googletest/* /usr/* ${CMAKE_SOURCE_DIR}/dependency/*)


set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
add_subdirectory(extern)
add_subdirectory(src)
enable_testing()
add_subdirectory(test)


