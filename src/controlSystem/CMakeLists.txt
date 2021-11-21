

add_library(controlSystem
        RunControl.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/RunControl.h
        RunConfiguration.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/RunConfiguration.h)

target_include_directories(controlSystem PRIVATE ${CMAKE_SOURCE_DIR}/include/)