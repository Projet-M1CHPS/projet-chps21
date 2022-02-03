

set(CURRENT_INCLUDE_DIR ${CURRENT_INCLUDE_DIR}/classifier)

add_library(Classifier STATIC
        classifierCollection.cpp ${CURRENT_INCLUDE_DIR}/classifierCollection.hpp
        classifierController.cpp ${CURRENT_INCLUDE_DIR}/classifierController.hpp
        classifierInputSet.cpp ${CURRENT_INCLUDE_DIR}/classifierInputSet.hpp
        classifierTracker.cpp ${CURRENT_INCLUDE_DIR}/classifierTracker.hpp
        classifierClass.cpp ${CURRENT_INCLUDE_DIR}/classifierClass.hpp
        )
target_include_directories(Classifier PUBLIC "${INCLUDE_DIR}" "${CURRENT_INCLUDE_DIR}")
target_link_libraries(Classifier PUBLIC Math Image OpenCL::OpenCL tscl::tscl clUtils)