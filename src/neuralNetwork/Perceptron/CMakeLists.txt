
set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/neuralNetwork/Perceptron)
add_library(Perceptron STATIC
        MLPerceptron.cpp ${CURRENT_INCLUDE_DIR}/MLPerceptron.hpp
        ${CURRENT_INCLUDE_DIR}/MLPSerializer.hpp
        Utf8MLPSerializer.cpp ${CURRENT_INCLUDE_DIR}/Utf8MLPSerializer.hpp
        ActivationFunction.cpp ${INCLUDE_DIR}/neuralNetwork/ActivationFunction.hpp
        )

target_include_directories(Perceptron PUBLIC ${INCLUDE_DIR} ${CURRENT_INCLUDE_DIR} ${INCLUDE_DIR}/neuralNetwork)
target_link_libraries(Perceptron PUBLIC tscl::tscl)