cmake_minimum_required(VERSION 3.16)

project(projet-chps21 VERSION 0.5.0.0 DESCRIPTION "Cancer detection using deep learning" LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)

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


if (CMAKE_COMPILER_IS_GNUCXX)
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
find_package(BLAS REQUIRED)

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
add_subdirectory(extern)
add_subdirectory(src)
enable_testing()
add_subdirectory(test)
add_subdirectory(apps)


