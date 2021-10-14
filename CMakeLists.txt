cmake_minimum_required(VERSION 3.20)
project(projet-chps21)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
add_subdirectory(src)

enable_testing()

add_subdirectory(test)

