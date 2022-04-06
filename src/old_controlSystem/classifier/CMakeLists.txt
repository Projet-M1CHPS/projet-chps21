

set(CURRENT_INCLUDE_DIR ${CURRENT_INCLUDE_DIR}/classifier)

add_library(Classifier STATIC
        CTCollection.cpp ${CURRENT_INCLUDE_DIR}/CTCollection.hpp
        CITCLoader.cpp ${CURRENT_INCLUDE_DIR}/CITCLoader.hpp
        CTController.cpp ${CURRENT_INCLUDE_DIR}/CTController.hpp
        CTrainingSet.cpp ${CURRENT_INCLUDE_DIR}/CTrainingSet.hpp
        CStatTracker.cpp ${CURRENT_INCLUDE_DIR}/CStatTracker.hpp
        ClassLabel.cpp ${CURRENT_INCLUDE_DIR}/ClassLabel.hpp
        )
target_include_directories(Classifier PUBLIC "${INCLUDE_DIR}" "${CURRENT_INCLUDE_DIR}")
target_link_libraries(Classifier PUBLIC Math Image OpenCL::OpenCL tscl::tscl clUtils NeuralNetwork)