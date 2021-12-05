

add_library(controlSystem
        controllerParameters.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/controllerParameters.hpp
        inputSetLoader.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/inputSetLoader.hpp
        inputSet.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/inputSet.hpp
        controller.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/controller.hpp
        )

target_include_directories(controlSystem PUBLIC "${CMAKE_SOURCE_DIR}/include")