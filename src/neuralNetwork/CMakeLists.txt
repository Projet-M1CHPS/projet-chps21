
add_subdirectory(Perceptron)
add_subdirectory(CNN)

set(CURRENT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/neuralNetwork)

# Create an interface library for the neural networks
add_library(nnet STATIC
        ${CURRENT_INCLUDE_DIR}/OptimizationScheduler.hpp
        ${CURRENT_INCLUDE_DIR}/OptimizerBatchScheduler.hpp
        SchedulerProfiler.cpp ${CURRENT_INCLUDE_DIR}/SchedulerProfiler.hpp
        ParallelScheduler.cpp ${CURRENT_INCLUDE_DIR}/ParallelScheduler.hpp
        SequentialScheduler.cpp ${CURRENT_INCLUDE_DIR}/SequentialScheduler.hpp
        ${INCLUDE_DIR}/NeuralNetwork.hpp
        )
target_include_directories(nnet PUBLIC "${INCLUDE_DIR}" "${CURRENT_INCLUDE_DIR}")
target_link_libraries(nnet PUBLIC m openclUtils Math Utils)

add_library(NeuralNetwork INTERFACE)
target_include_directories(NeuralNetwork INTERFACE "${INCLUDE_DIR}" "${CURRENT_INCLUDE_DIR}")
target_link_libraries(NeuralNetwork INTERFACE m Utils nnet Perceptron CNN)