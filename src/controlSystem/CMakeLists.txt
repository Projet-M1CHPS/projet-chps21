

set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/controlSystem)
add_library(ControlSystem
        ControllerResult.cpp ${CURRENT_INCLUDE_DIR}/ControllerResult.hpp
        TrainingController.cpp ${CURRENT_INCLUDE_DIR}/TrainingController.hpp
        EvalController.cpp ${CURRENT_INCLUDE_DIR}/EvalController.hpp
        InputSet.cpp ${CURRENT_INCLUDE_DIR}/InputSet.hpp
        TrainingCollection.cpp ${CURRENT_INCLUDE_DIR}/TrainingCollection.hpp
        TrainingCollectionLoader.cpp ${CURRENT_INCLUDE_DIR}/TrainingCollectionLoader.hpp
        InputSetLoader.cpp ${CURRENT_INCLUDE_DIR}/InputSetLoader.hpp ModelEvaluator.cpp ${CURRENT_INCLUDE_DIR}/ModelEvaluator.hpp)

target_link_libraries(ControlSystem PUBLIC NeuralNetwork openclUtils OpenMP::OpenMP_CXX)
target_include_directories(ControlSystem PUBLIC ${CURRENT_INCLUDE_DIR})