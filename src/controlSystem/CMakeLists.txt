

add_library(controlSystem
        controllerParameters.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/controllerParameters.hpp
        setLoader.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/setLoader.hpp
        inputSet.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/inputSet.hpp)

target_include_directories(controlSystem PUBLIC "${CMAKE_SOURCE_DIR}/include")