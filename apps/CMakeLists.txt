
# Test app
add_executable(main main.cpp)
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(main PUBLIC "${INCLUDE_DIR}")
target_link_libraries(main PRIVATE ControlSystem Image NeuralNetwork Threads::Threads tscl::tscl ${BLAS_LIBRARIES})


# MPI app
if (USE_MPI)
    message(STATUS "Using MPI")
    add_executable(mpi_main mpi_main.cpp)
    set_target_properties(mpi_main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    target_include_directories(mpi_main PUBLIC "${INCLUDE_DIR}")
    target_link_libraries(mpi_main PRIVATE MPIControlSystem Image MPINeuralNetwork Threads::Threads tscl::tscl ${BLAS_LIBRARIES})
else ()
    message(STATUS "MPI not found")
endif ()

add_executable(gemm_benchmark benchmark/gemm_benchmark.cpp)
set_target_properties(gemm_benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(gemm_benchmark PUBLIC "${INCLUDE_DIR}")
target_link_libraries(gemm_benchmark PRIVATE openclUtils NeuralNetwork Threads::Threads tscl::tscl ${BLAS_LIBRARIES})

add_executable(convo convo.cpp)
set_target_properties(convo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_include_directories(convo PUBLIC "${INCLUDE_DIR}")
target_link_libraries(convo PRIVATE NeuralNetwork Threads::Threads tscl::tscl openclUtils ${BLAS_LIBRARIES})