
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
        BLAS::BLAS
)
gtest_discover_tests(NeuralNetwork_test)