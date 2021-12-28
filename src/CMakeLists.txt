
add_library(Utils STATIC
        Utils.cpp ${CMAKE_SOURCE_DIR}/include/Utils.hpp
        )
set_target_properties(Utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(Utils PUBLIC "${CMAKE_SOURCE_DIR}/include")

# This avoids adding the include path to all targets that needs Matrix.h
# It also propagates the dependency on BLAS
add_library(Math INTERFACE)
target_include_directories(Math INTERFACE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(Math INTERFACE BLAS::BLAS)

add_subdirectory(image)
add_subdirectory(controlSystem)
add_subdirectory(neuralNetwork)
