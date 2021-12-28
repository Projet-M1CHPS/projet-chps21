

add_subdirectory(Perceptron)
add_subdirectory(Model)

set(CURRENT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/neuralNetwork)
add_library(NeuralNetwork INTERFACE)
target_include_directories(NeuralNetwork INTERFACE "${INCLUDE_DIR}" "${CURRENT_INCLUDE_DIR}")
target_link_libraries(NeuralNetwork INTERFACE m Utils Model)
