

set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/controlSystem)

add_subdirectory(classifier)

add_library(ControlSystem STATIC
        controllerParameters.cpp ${CURRENT_INCLUDE_DIR}/controllerParameters.hpp
        inputSet.cpp ${CURRENT_INCLUDE_DIR}/inputSet.hpp
        controller.cpp ${CURRENT_INCLUDE_DIR}/controller.hpp
        imageInputSet.cpp ${CURRENT_INCLUDE_DIR}/imageInputSet.hpp
        ${INCLUDE_DIR}/Control.hpp
        )

target_include_directories(ControlSystem PUBLIC "${INCLUDE_DIR}" "${CURRENT_INCLUDE_DIR}")
target_link_libraries(ControlSystem PUBLIC clUtils Math Image tscl::tscl NeuralNetwork)