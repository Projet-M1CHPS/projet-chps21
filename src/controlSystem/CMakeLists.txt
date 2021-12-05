

add_library(ControlSystem
        controllerParameters.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/controllerParameters.hpp
        inputSetLoader.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/inputSetLoader.hpp
        inputSet.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/inputSet.hpp
        controller.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/controller.hpp
        )

add_subdirectory(classifier)

target_include_directories(ControlSystem PUBLIC "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/include/controlSystem")