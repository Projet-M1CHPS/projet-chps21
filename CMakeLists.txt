cmake_minimum_required(VERSION 3.16)

project(projet-chps21 VERSION 1.1.0.0 DESCRIPTION "Cancer detection using deep learning" LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Setup compiler specific flags
include(CheckCXXCompilerFlag)
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fast -O3 -qmkl -lm")
  # Intel compilers are not supported for coverage tests
  set(CODE_COVERAGE OFF)
else ()
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

  if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  endif ()
endif ()

include(check_coverage)
maybe_enable_coverage()

# Defines version specific Macro
configure_file(include/ProjectConstant.hpp.in include/ProjectVersion.hpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)


# Comment one out to choose which mkl to use
# Intel's mkl sequential implementation
set(BLA_VENDOR "Intel10_64ilp_seq")
# Intel's mkl parallel implementation
# set(BLA_VENDOR "Intel10_64ilp")


find_package(Threads REQUIRED)
find_package(OpenCL REQUIRED)
find_package(CLBlast REQUIRED)

find_package(BLAS REQUIRED)

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
# Setup external dependency of the project
add_subdirectory(extern)
# Main project directory
add_subdirectory(src)

# ! Must be called in the root cmakelist !
# otherwise, the test target will not be properly set up
enable_testing()
add_subdirectory(test)

# Contains various end executable
add_subdirectory(apps)

# Documentation generator setup
# Uses Doxygen
# (Optionnal)
add_subdirectory(docs)


