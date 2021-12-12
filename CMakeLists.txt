cmake_minimum_required(VERSION 3.16)
project(projet-chps21 VERSION 0.3.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

if (COMPILER_SUPPORTS_MARCH_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fast -qmkl")
  # Intel compilers are not supported for coverage tests
  set(CODE_COVERAGE OFF)
  message(STATUS ${CMAKE_CXX_FLAGS})
endif ()

include(dependency/cmake_coverage.cmake)
configure_file(include/tscl/ProjectConstant.hpp.in include/ProjectVersion.hpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# We remove googletest and the std from the coverage report
# We also remove the test directory, since templated functions that are used
# will appear in their respective headers
add_code_coverage_all_targets(EXCLUDE ${CMAKE_SOURCE_DIR}/test/* */googletest/* /usr/* ${CMAKE_SOURCE_DIR}/dependency/*)

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
add_subdirectory(src)

enable_testing()

add_subdirectory(test)

