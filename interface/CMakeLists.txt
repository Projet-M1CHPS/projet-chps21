## Extern tools
# JSON parsing lib:
include(nlohmann_json.cmake)
# Pythin binding lib:
include(pybind11.cmake)

## Interface build
add_subdirectory(cpp_lib_interface)


file(GLOB PYTHON_FILES "${CMAKE_SOURCE_DIR}/interface/cpp_lib_interface/*.cpp" "${CMAKE_SOURCE_DIR}/interface/cpp_lib_interface/*.hpp")
pybind11_add_module(kreps
        ${SOURCE_DIR}
        ${INCLUDE_DIR}
        ${PYTHON_FILES}
        )
target_link_libraries(kreps PUBLIC NetworkInterface nlohmann_json::nlohmann_json ControlSystem NeuralNetwork Threads::Threads tscl::tscl BLAS::BLAS)

install(SCRIPT kreps
        RUNTIME DESTINATION "${CMAKE_BINARY_DIR}/bin"
        )
configure_file(main.py "${CMAKE_BINARY_DIR}/interface" COPYONLY)

