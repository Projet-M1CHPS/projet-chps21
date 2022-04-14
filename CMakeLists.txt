cmake_minimum_required(VERSION 3.16)

project(projet-chps21 VERSION 1.2.1.0 DESCRIPTION "Cancer detection using deep learning" LANGUAGES CXX C)

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
if (COVERARE_ENABLED)
    # Must be placed at the top level for compiler flags to propagate to all targets
    append_coverage_compiler_flags()
endif ()

# Defines version specific Macro
configure_file(include/ProjectConstant.hpp.in include/ProjectVersion.hpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)


# Comment one out to choose which mkl to use
# Intel's mkl sequential implementation
# set(BLA_VENDOR "Intel10_64ilp_seq")
# Intel's mkl parallel implementation
# set(BLA_VENDOR "Intel10_64ilp")


find_package(Threads REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
# Required for std::algo
find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

find_package(CLBlast REQUIRED)

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
# Setup external dependency of the project
add_subdirectory(extern)

# Messy way to setup extra error messages, but i don't want to spend time on it
CHECK_CXX_COMPILER_FLAG("-Wall -Wpedantic -Wextra -Wno-sign-compare" COMPILER_SUPPORT_WALL_WERROR_WPEDANTIC_WEXTRA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Wno-sign-compare")

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


