

set(CURRENT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/neuralNetwork/Model)
add_library(Model
        MLPModel.cpp ${CURRENT_INCLUDE_DIR}/MLPModel.hpp
        MLPModelOptimizer.cpp ${CURRENT_INCLUDE_DIR}/MLPModelOptimizer.hpp
        Utf8MLPModelSerializer.cpp ${CURRENT_INCLUDE_DIR}/MLPModelSerializer.hpp
        Utf8MLPModelSerializer.cpp ${CURRENT_INCLUDE_DIR}/Utf8MLPModelSerializer.hpp
        )
target_link_libraries(Model Perceptron)
target_include_directories(Model PUBLIC ${CURRENT_INCLUDE_DIR})