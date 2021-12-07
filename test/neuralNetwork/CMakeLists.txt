
add_executable(
        NeuralNetwork_test
        NeuralNetwork_test.cpp
        ActivationFunction_test.cpp
)

target_link_libraries(
        NeuralNetwork_test PUBLIC
        NeuralNetwork
        Utils
        gtest_main
        ${BLAS_LIBRARIES}
)

target_code_coverage(NeuralNetwork_test ALL)

gtest_discover_tests(NeuralNetwork_test)