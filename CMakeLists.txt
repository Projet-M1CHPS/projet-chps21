cmake_minimum_required(VERSION 3.16)
project(projet-chps21)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Enable for gprof
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -march=native")

include(dependency/cmake_coverage.cmake)

# We remove googletest and the std from the coverage report
# We also remove the test directory, since templated functions that are used
# will appear in their respective headers
add_code_coverage_all_targets(EXCLUDE ${CMAKE_SOURCE_DIR}/test/* */googletest/* /usr/* ${CMAKE_SOURCE_DIR}/dependency/*)

include_directories(include)    
add_subdirectory(src)

enable_testing()

add_subdirectory(test)

