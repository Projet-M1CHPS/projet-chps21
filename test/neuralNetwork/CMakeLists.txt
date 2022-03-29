
add_executable(
        NeuralNetwork_test
        NeuralNetwork_test.cpp
        ActivationFunction_test.cpp
)

target_link_libraries(
        NeuralNetwork_test PUBLIC
        Utils
        gtest_main
        NeuralNetwork
        BLAS::BLAS
)
gtest_discover_tests(NeuralNetwork_test)