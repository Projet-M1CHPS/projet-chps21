
# Test app
add_executable(main main.cpp)
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(main PUBLIC "${INCLUDE_DIR}")
target_link_libraries(main PRIVATE ControlSystem2 Image NeuralNetwork Threads::Threads tscl::tscl BLAS::BLAS)

# MPI app
if (USE_MPI)
    message(STATUS "Using MPI")
    add_executable(mpi_main mpi_main.cpp)
    set_target_properties(mpi_main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    target_include_directories(mpi_main PUBLIC "${INCLUDE_DIR}")
    target_link_libraries(mpi_main PRIVATE MPIControlSystem Image NeuralNetwork Threads::Threads tscl::tscl BLAS::BLAS)
else ()
    message(STATUS "MPI not found")
endif ()


# Test app
add_executable(xor xor.cpp)
set_target_properties(xor PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(xor PUBLIC "${INCLUDE_DIR}")
target_link_libraries(xor PRIVATE NeuralNetwork Threads::Threads tscl::tscl BLAS::BLAS clUtils)

add_executable(gemm_benchmark benchmark/gemm_benchmark.cpp)
set_target_properties(gemm_benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(gemm_benchmark PUBLIC "${INCLUDE_DIR}")
target_link_libraries(gemm_benchmark PRIVATE openclUtils NeuralNetwork Threads::Threads tscl::tscl BLAS::BLAS)

add_executable(convo convo.cpp)
set_target_properties(convo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(convo PUBLIC "${INCLUDE_DIR}")
target_link_libraries(convo PRIVATE NeuralNetwork Threads::Threads tscl::tscl BLAS::BLAS clUtils)