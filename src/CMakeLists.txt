
set(BLA_VENDOR)
find_package(BLAS REQUIRED)

add_library(Utils STATIC
        Utils.cpp ${CMAKE_SOURCE_DIR}/include/Utils.hpp
        )
set_target_properties(Utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(Utils PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_code_coverage(Utils)

add_subdirectory(tscl)
add_subdirectory(image)
add_subdirectory(controlSystem)
add_subdirectory(neuralNetwork)

add_executable(main main.cpp)
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(main PUBLIC "${INCLUDE_DR}")

target_link_libraries(main PRIVATE ControlSystem Image NeuralNetwork pthread tscl BLAS::BLAS)