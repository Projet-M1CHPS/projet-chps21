
# Test app
add_executable(main main.cpp)
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(main PUBLIC "${INCLUDE_DIR}")
target_link_libraries(main PRIVATE ControlSystem Image NeuralNetwork Threads::Threads tscl::tscl BLAS::BLAS)