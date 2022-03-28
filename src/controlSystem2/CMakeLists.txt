

set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/controlSystem2)
add_library(controlSystem2
        Controller.cpp ${CURRENT_INCLUDE_DIR}/Controller.hpp
        TrainingController.cpp ${CURRENT_INCLUDE_DIR}/TrainingController.hpp
        EvalController.cpp ${CURRENT_INCLUDE_DIR}/EvalController.hpp
        InputSet.cpp ${CURRENT_INCLUDE_DIR}/InputSet.hpp
        TrainingCollection.cpp ../../include/controlSystem2/TrainingCollection.hpp)

target_link_libraries(controlSystem2 PUBLIC NeuralNetwork)
target_include_directories(controlSystem2 PUBLIC ${CURRENT_INCLUDE_DIR})