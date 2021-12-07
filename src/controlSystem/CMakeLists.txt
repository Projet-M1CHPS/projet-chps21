

set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/controlSystem)
set(CLASSIFIER_INCLUDE_DIR ${CURRENT_INCLUDE_DIR}/classifier)

add_library(ControlSystem
        controllerParameters.cpp ${CURRENT_INCLUDE_DIR}/controllerParameters.hpp
        inputSetLoader.cpp ${CURRENT_INCLUDE_DIR}/inputSetLoader.hpp
        inputSet.cpp ${CURRENT_INCLUDE_DIR}/inputSet.hpp
        controller.cpp ${CURRENT_INCLUDE_DIR}/controller.hpp
        classifier/classifierCollection.cpp ${CLASSIFIER_INCLUDE_DIR}/classifierCollection.hpp
        classifier/classifierController.cpp ${CLASSIFIER_INCLUDE_DIR}/classifierController.hpp
        classifier/classifierInputSet.cpp ${CLASSIFIER_INCLUDE_DIR}/classifierInputSet.hpp
        classifier/classifierTracker.cpp ${CLASSIFIER_INCLUDE_DIR}/classifierTracker.hpp
        )

target_include_directories(ControlSystem PUBLIC "${INCLUDE_DIR}" "${CURRENT_INCLUDE_DIR}" "${CLASSIFIER_INCLUDE_DIR}")
target_code_coverage(ControlSystem)