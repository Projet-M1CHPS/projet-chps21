
set(BLA_VENDOR)
find_package(BLAS REQUIRED)
if (BLAS_FOUND)
  include_directories(/opt/OpenBLAS/include/)
endif (BLAS_FOUND)

add_library(Utils STATIC
        Utils.cpp ${CMAKE_SOURCE_DIR}/include/Utils.hpp
        )
set_target_properties(Utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(Utils PUBLIC "${CMAKE_SOURCE_DIR}/include" ${BLAS_INCLUDE_DIRS})
target_code_coverage(Utils)

add_subdirectory(tscl)
add_subdirectory(image)
add_subdirectory(controlSystem)
add_subdirectory(NeuralNetwork)

add_executable(main main.cpp)
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(main PUBLIC "${INCLUDE_DR}")

target_link_libraries(main PRIVATE ControlSystem Image NeuralNetwork pthread tscl ${BLAS_LIBRARIES})