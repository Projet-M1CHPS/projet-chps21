include(nolhmannjson.cmake)
include(pybind11.cmake)

## Interface build

file (GLOB PYTHON_FILES "${CMAKE_SOURCE_DIR}/interface/*.cpp" "${CMAKE_SOURCE_DIR}/interface/*.hpp")

pybind11_add_module(kreps 
	${SOURCE_DIR}
	${INCLUDE_DIR}
	${PYTHON_FILES}
)

add_library(NetworkInterface INTERFACE NetworkInterface.cpp)
target_include_directories(NetworkInterface INTERFACE "${CMAKE_SOURCE_DIR}/interface")

target_link_libraries(kreps PUBLIC NetworkInterface nlohmann_json::nlohmann_json ControlSystem NeuralNetwork Threads::Threads tscl::tscl BLAS::BLAS)

install(SCRIPT kreps 
RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin"
)
configure_file(main.py "${CMAKE_BINARY_DIR}/interface" COPYONLY) 