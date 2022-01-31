
set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/neuralNetwork/CNN)
add_library(CNN STATIC
        ConvolutionLayer.cpp ${CURRENT_INCLUDE_DIR}/ConvolutionLayer.hpp
        PoolingLayer.cpp ${CURRENT_INCLUDE_DIR}/PoolingLayer.hpp
        Filter.cpp ${CURRENT_INCLUDE_DIR}/Filter.hpp
        )

target_include_directories(CNN PUBLIC ${INCLUDE_DIR} ${CURRENT_INCLUDE_DIR} ${INCLUDE_DIR}/neuralNetwork)
target_link_libraries(CNN PUBLIC tscl::tscl)