
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/e2239ee6043f73722e7aa812a459f54a28552929.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(BLAS REQUIRED)

add_executable(
        Math_test
        Math_test.cpp
        ${CMAKE_SOURCE_DIR}/include/Matrix.hpp
)

target_link_libraries(
        Math_test PUBLIC
        Utils
        gtest_main
        BLAS::BLAS
)

target_code_coverage(Math_test ALL)

include(GoogleTest)
add_subdirectory(image)
add_subdirectory(controlSystem)
add_subdirectory(neuralNetwork)

gtest_discover_tests(Math_test)

