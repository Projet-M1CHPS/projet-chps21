
set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/neuralNetwork/CNN)
add_library(CNN STATIC
        CNNTopology.cpp ${CURRENT_INCLUDE_DIR}/CNNTopology.hpp
        Filter.cpp ${CURRENT_INCLUDE_DIR}/Filter.hpp
        CNNLayer.cpp ${CURRENT_INCLUDE_DIR}/CNNLayer.hpp
        CNN.cpp ${CURRENT_INCLUDE_DIR}/CNN.hpp
        CNNStorageBP.cpp ${CURRENT_INCLUDE_DIR}/CNNStorageBP.hpp
        )

target_include_directories(CNN PUBLIC ${INCLUDE_DIR} ${CURRENT_INCLUDE_DIR} ${INCLUDE_DIR}/neuralNetwork)
target_link_libraries(CNN PUBLIC tscl::tscl)