
add_library(NetworkInterface STATIC NetworkInterface.cpp NetworkInterface.hpp)
set_target_properties(NetworkInterface PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(NetworkInterface PUBLIC "${CMAKE_SOURCE_DIR}/interface/cpp_lib_interface")
target_link_libraries(NetworkInterface PUBLIC nlohmann_json::nlohmann_json ControlSystem Image NeuralNetwork Threads::Threads tscl::tscl BLAS::BLAS)

