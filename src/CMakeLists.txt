

add_library(Utils STATIC
        Utils.cpp ${CMAKE_SOURCE_DIR}/include/Utils.hpp
        )
set_target_properties(Utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(Utils PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_code_coverage(Utils)

add_library(Image STATIC
        Image.cpp ${CMAKE_SOURCE_DIR}/include/Image.hpp
        ${CMAKE_SOURCE_DIR}/dependency/stb_image.h
        ${CMAKE_SOURCE_DIR}/dependency/stb_image_write.h
        )
set_target_properties(Image PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
# Uses the stb library located in dependency
target_include_directories(Image PUBLIC "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/dependency")
target_code_coverage(Image)

add_library(Transform STATIC
        TransformEngine.cpp
        Transform.cpp ${CMAKE_SOURCE_DIR}/include/Transform.hpp
        )
set_target_properties(Transform PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(Transform PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_code_coverage(Transform)


set(BLA_VENDER OpenBLAS)
find_package(BLAS REQUIRED)
add_library(NeuralNetwork STATIC
        NeuralNetwork.cpp ${CMAKE_SOURCE_DIR}/include/NeuralNetwork.hpp
        ActivationFunction.cpp ${CMAKE_SOURCE_DIR}/include/ActivationFunction.hpp
        NeuralNetworkSerializer.cpp ${CMAKE_SOURCE_DIR}/include/NeuralNetworkSerializer.hpp
        ${CMAKE_SOURCE_DIR}/include/Matrix.hpp
        )
set_target_properties(NeuralNetwork PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(NeuralNetwork PUBLIC "${CMAKE_SOURCE_DIR}/include" ${BLAS_INCLUDE_DIRS})
target_link_libraries(NeuralNetwork PUBLIC m Utils ${BLAS_LIBRARIES})
target_code_coverage(NeuralNetwork)

add_executable(main main.cpp)
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(main PUBLIC "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(main PRIVATE Image Transform NeuralNetwork pthread)