

add_subdirectory(Perceptron)

set(CURRENT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/neuralNetwork)
add_library(NeuralNetwork STATIC
        ActivationFunction.cpp ${CURRENT_INCLUDE_DIR}/ActivationFunction.hpp
        ModelOptimizer.cpp ${CURRENT_INCLUDE_DIR}/ModelOptimizer.hpp
        OptimizerFactory.cpp ${CURRENT_INCLUDE_DIR}/OptimizerFactory.hpp
        ${INCLUDE_DIR}/Matrix.hpp
        ${INCLUDE_DIR}/Network.hpp
        )
set_target_properties(NeuralNetwork PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(NeuralNetwork PUBLIC "${INCLUDE_DIR}" "${CURRENT_INCLUDE_DIR}")
target_link_libraries(NeuralNetwork PUBLIC m Utils Perceptron)
target_code_coverage(NeuralNetwork)
