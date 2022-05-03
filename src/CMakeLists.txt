
add_library(Utils STATIC
        Utils.cpp ${CMAKE_SOURCE_DIR}/include/Utils.hpp
        )
set_target_properties(Utils PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(Utils PUBLIC "${CMAKE_SOURCE_DIR}/include")

add_subdirectory(math)
add_subdirectory(openclUtils)
add_subdirectory(image)
add_subdirectory(controlSystem)
if (USE_MPI)
    add_subdirectory(mpiWrapper)
endif ()
add_subdirectory(neuralNetwork)
add_subdirectory(kernels)
