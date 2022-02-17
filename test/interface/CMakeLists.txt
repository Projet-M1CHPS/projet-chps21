
add_executable(Interface_cpp_test Interface_cpp_test.cpp)

target_link_libraries(
        Interface_cpp_test PUBLIC
        NetworkInterface
        nlohmann_json::nlohmann_json ControlSystem Image NeuralNetwork Threads::Threads tscl::tscl BLAS::BLAS
        gtest_main
)

gtest_discover_tests(Interface_cpp_test)