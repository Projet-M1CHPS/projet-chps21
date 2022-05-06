
set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/neuralNetwork/CNN)
add_library(CNN STATIC
        CNN.cpp ${CURRENT_INCLUDE_DIR}/CNN.hpp
        CNNTopologyLayer.cpp ${CURRENT_INCLUDE_DIR}/CNNTopologyLayer.hpp
        CNNTopology.cpp ${CURRENT_INCLUDE_DIR}/CNNTopology.hpp

        CNNModel.cpp ${CURRENT_INCLUDE_DIR}/CNNModel.hpp

        CNNLayer.cpp ${CURRENT_INCLUDE_DIR}/CNNLayer.hpp
        CNNStorageBP.cpp ${CURRENT_INCLUDE_DIR}/CNNStorageBP.hpp

        CNNOptimizer.cpp ${CURRENT_INCLUDE_DIR}/CNNOptimizer.hpp

        CNNOptimization/CNNSGDOptimization.cpp ${CURRENT_INCLUDE_DIR}/CNNOptimization/CNNSGDOptimization.hpp
        )

target_include_directories(CNN PUBLIC ${INCLUDE_DIR} ${CURRENT_INCLUDE_DIR} ${INCLUDE_DIR}/neuralNetwork ${CURRENT_INCLUDE_DIR}/CNNOptimization)
target_link_libraries(CNN PUBLIC openclUtils tscl::tscl Perceptron)