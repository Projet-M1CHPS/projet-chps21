

set(CURRENT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/neuralNetwork/Model)
add_library(Model
        MLPModel.cpp ${CURRENT_INCLUDE_DIR}/MLPModel.hpp
        MLPModelOptimizer.cpp ${CURRENT_INCLUDE_DIR}/MLPModelOptimizer.hpp
        PlainTextMLPModelSerializer.cpp ${CURRENT_INCLUDE_DIR}/MLPModelSerializer.hpp
        PlainTextMLPModelSerializer.cpp ${CURRENT_INCLUDE_DIR}/PlainTextMLPModelSerializer.hpp

        CNNModel.cpp ${CURRENT_INCLUDE_DIR}/CNNModel.hpp
        CNNModelOptimizer.cpp ${CURRENT_INCLUDE_DIR}/CNNModelOptimizer.hpp
        )
target_link_libraries(Model Perceptron CNN)
target_include_directories(Model PUBLIC ${CURRENT_INCLUDE_DIR})