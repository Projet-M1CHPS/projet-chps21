
set(BLA_VENDOR OpenBLAS)
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


add_library(Image STATIC
        Image.cpp ${CMAKE_SOURCE_DIR}/include/Image.hpp
        TransformEngine.cpp
        Transform.cpp ${CMAKE_SOURCE_DIR}/include/Transform.hpp
        ${CMAKE_SOURCE_DIR}/dependency/stb_image.h
        ${CMAKE_SOURCE_DIR}/dependency/stb_image_write.h
        )
set_target_properties(Image PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
# Uses the stb library located in dependency
target_include_directories(Image PUBLIC "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/dependency")
target_code_coverage(Image)

add_subdirectory(controlSystem)
add_subdirectory(NeuralNetwork)

add_executable(main main.cpp)
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(main PUBLIC "${INCLUDE_DR}")

target_link_libraries(main PRIVATE ControlSystem Image NeuralNetwork pthread ${BLAS_LIBRARIES})