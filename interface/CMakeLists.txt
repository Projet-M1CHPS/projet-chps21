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

set_property(TARGET tscl PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(kreps PUBLIC NetworkInterface nlohmann_json::nlohmann_json ControlSystem NeuralNetwork Threads::Threads tscl::tscl BLAS::BLAS)

message("= = = = = = = = = = = = = =")
set(PYTHON_LIBRARY_DIR "/usr/lib/python3.8/")
message(${PYTHON_LIBRARY_DIR})
set(PYTHON_EXECUTABLE "/usr/bin/python3.8")
message(${PYTHON_EXECUTABLE})
message("===========================")

install(SCRIPT kreps 
RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin"
)
configure_file(main.py "${CMAKE_BINARY_DIR}/interface" COPYONLY) 