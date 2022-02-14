
set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/neuralNetwork/Perceptron)
add_library(Perceptron STATIC
        MLPerceptron.cpp ${CURRENT_INCLUDE_DIR}/MLPerceptron.hpp
        MLPSerializer.cpp ${CURRENT_INCLUDE_DIR}/MLPSerializer.hpp

        MLPModel.cpp ${CURRENT_INCLUDE_DIR}/MLPModel.hpp
        MLPModelSerializer.cpp ${CURRENT_INCLUDE_DIR}/MLPModelSerializer.hpp

        Optimization/Optimization.cpp ${CURRENT_INCLUDE_DIR}/Optimization/Optimization.hpp
        MLPOptimizer.cpp ${CURRENT_INCLUDE_DIR}/MLPOptimizer.hpp
        )

target_include_directories(Perceptron PUBLIC ${INCLUDE_DIR} ${CURRENT_INCLUDE_DIR} ${INCLUDE_DIR}/neuralNetwork)
target_link_libraries(Perceptron PUBLIC tscl::tscl)