
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/e2239ee6043f73722e7aa812a459f54a28552929.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
        Math_test
        Math_test.cpp
        ${CMAKE_SOURCE_DIR}/include/Matrix.hpp
)

target_link_libraries(
        Math_test PUBLIC
        Utils
        gtest_main
)

target_code_coverage(Math_test ALL)

add_executable(
        NeuralNetwork_test
        NeuralNetwork_test.cpp
)

target_link_libraries(
        NeuralNetwork_test PUBLIC
        NeuralNetwork
        Utils
        gtest_main
)

target_code_coverage(NeuralNetwork_test ALL)

add_executable(
        Image_test
        Image_test.cpp
)

target_link_libraries(
        Image_test PUBLIC
        Image
        gtest_main
)

target_code_coverage(Image_test ALL)

add_executable(
        Transform_test
        Transform_test.cpp
)

target_link_libraries(
        Transform_test PUBLIC
        Transform
        Image
        gtest_main
)

target_code_coverage(Transform_test ALL)


add_executable(
        ActivationFunction_test
        ActivationFunction_test.cpp
)

target_link_libraries(
        ActivationFunction_test PUBLIC
        Utils
        gtest_main
)

target_code_coverage(ActivationFunction_test ALL)


include(GoogleTest)
gtest_discover_tests(Math_test)
gtest_discover_tests(NeuralNetwork_test)

gtest_discover_tests(ActivationFunction_test)
gtest_discover_tests(Image_test)
gtest_discover_tests(Transform_test)
