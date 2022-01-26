cmake_minimum_required(VERSION 3.16)

project(projet-chps21 VERSION 0.5.0.2 DESCRIPTION "Cancer detection using deep learning" LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Compiler specific flags
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fast -O3 -qmkl -lm")
  # Intel compilers are not supported for coverage tests
  set(CODE_COVERAGE OFF)
  message(STATUS ${CMAKE_CXX_FLAGS_RELEASE})
else ()
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

  if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  endif ()
endif ()


if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  # Code coverage flags
  # Added as an external plugin
  include(extern/CodeCoverage.cmake)
  append_coverage_compiler_Flags()
  set(COVERAGE_EXCLUDES "/extern" "/test")
endif ()

# Defines version specific Macro
configure_file(include/ProjectConstant.hpp.in include/ProjectVersion.hpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

find_package(Threads REQUIRED)

if (CMAKE_COMPILER_IS_INTELCXX)
  # Comment one out to choose which mkl to use
  # Intel's mkl sequential implementation
  # set(BLA_VENDOR "Intel10_64ilp_seq")
  # Intel's mkl parallel implementation
  set(BLA_VENDOR "Intel10_64ilp")
endif ()

find_package(BLAS REQUIRED)

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
add_subdirectory(extern)
add_subdirectory(src)
enable_testing()
add_subdirectory(test)
add_subdirectory(apps)
add_subdirectory(docs)


