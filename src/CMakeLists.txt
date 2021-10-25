

add_library(Image STATIC Image.cpp ${CMAKE_SOURCE_DIR}/include/Image.hpp)
set_target_properties(Image PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(Image PUBLIC "${CMAKE_SOURCE_DIR}/include")

add_library(NeuralNetwork STATIC NeuralNetwork.cpp ${CMAKE_SOURCE_DIR}/include/NeuralNetwork.hpp)
set_target_properties(NeuralNetwork PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(NeuralNetwork PUBLIC "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(NeuralNetwork m)

add_executable(main main.cpp)
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(main PUBLIC "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(main Image NeuralNetwork)