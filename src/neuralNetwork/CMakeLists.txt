
add_subdirectory(Perceptron)

set(CURRENT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/neuralNetwork)

# Create an interface library for the neural networks
add_library(NeuralNetwork INTERFACE)
target_include_directories(NeuralNetwork INTERFACE "${INCLUDE_DIR}" "${CURRENT_INCLUDE_DIR}")
target_link_libraries(NeuralNetwork INTERFACE m Utils Perceptron)
