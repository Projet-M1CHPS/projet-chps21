
set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/neuralNetwork/CNN)
add_library(CNN STATIC
        CNN.cpp ${CURRENT_INCLUDE_DIR}/CNN.hpp
        CNNTopology.cpp ${CURRENT_INCLUDE_DIR}/CNNTopology.hpp

        CNNModel.cpp ${CURRENT_INCLUDE_DIR}/CNNModel.hpp

        Filter.cpp ${CURRENT_INCLUDE_DIR}/Filter.hpp
        CNNLayer.cpp ${CURRENT_INCLUDE_DIR}/CNNLayer.hpp
        CNNStorageBP.cpp ${CURRENT_INCLUDE_DIR}/CNNStorageBP.hpp

        CNNOptimizer.cpp ${CURRENT_INCLUDE_DIR}/CNNOptimizer.hpp
        ../ActivationFunction.cpp ${INCLUDE_DIR}/neuralNetwork/ActivationFunction.hpp
        )

target_include_directories(CNN PUBLIC ${INCLUDE_DIR} ${CURRENT_INCLUDE_DIR} ${INCLUDE_DIR}/neuralNetwork)
target_link_libraries(CNN PUBLIC clUtils tscl::tscl Perceptron)