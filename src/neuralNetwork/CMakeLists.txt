set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/neuralNetwork)

add_library(nnet INTERFACE)
target_include_directories(nnet INTERFACE "${INCLUDE_DIR}" "${CURRENT_INCLUDE_DIR}")
target_link_libraries(nnet INTERFACE m openclUtils Math Utils)

add_subdirectory(Perceptron)
add_subdirectory(CNN)
add_subdirectory(OptimizationScheduler)

add_library(NeuralNetwork INTERFACE)
target_include_directories(NeuralNetwork INTERFACE "${INCLUDE_DIR}" "${CURRENT_INCLUDE_DIR}")
target_link_libraries(NeuralNetwork INTERFACE m Utils nnet Perceptron CNN OptimizationScheduler)

if (USE_MPI)
    add_library(MPINeuralNetwork INTERFACE)
    target_include_directories(MPINeuralNetwork INTERFACE "${INCLUDE_DIR}" "${CURRENT_INCLUDE_DIR}")
    target_link_libraries(MPINeuralNetwork INTERFACE m Utils nnet MPIPerceptron CNN OptimizationScheduler)
endif ()