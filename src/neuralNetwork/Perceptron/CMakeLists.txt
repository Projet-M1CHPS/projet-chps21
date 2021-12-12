
set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/neuralNetwork/Perceptron)
add_library(Perceptron
        MLPerceptron.cpp ${CURRENT_INCLUDE_DIR}/MLPerceptron.hpp
        MLPOptimizer.cpp ${CURRENT_INCLUDE_DIR}/MLPOptimizer.hpp
        MLPSerializer.cpp ${CURRENT_INCLUDE_DIR}/MLPSerializer.hpp
        MLPModel.cpp ${CURRENT_INCLUDE_DIR}/MLPModel.hpp
        )
target_include_directories(Perceptron PUBLIC ${INCLUDE_DIR} ${CURRENT_INCLUDE_DIR} ${INCLUDE_DIR}/neuralNetwork)