
add_subdirectory(Perceptron)
add_subdirectory(CNN)

set(CURRENT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/neuralNetwork)

# Create an interface library for the neural networks
add_library(NeuralNetwork BatchScheduler.cpp ${CURRENT_INCLUDE_DIR}/BatchScheduler.hpp ${INCLUDE_DIR}/NeuralNetwork.hpp)
target_include_directories(NeuralNetwork PUBLIC "${INCLUDE_DIR}" "${CURRENT_INCLUDE_DIR}")
target_link_libraries(NeuralNetwork PUBLIC m clUtils Utils)
