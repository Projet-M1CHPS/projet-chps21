

add_library(controlSystem
        RunControl.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/RunControl.hpp
        RunConfiguration.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/RunConfiguration.hpp
        ImageCache.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/ImageCache.hpp)

target_include_directories(controlSystem PRIVATE ${CMAKE_SOURCE_DIR}/include/ ${BLAS_INCLUDE_DIRS})