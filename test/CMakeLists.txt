
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/e2239ee6043f73722e7aa812a459f54a28552929.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(
        Math_test
        Math_test.cpp
        ${CMAKE_SOURCE_DIR}/include/Matrix.hpp
        clFMatrix_test.cpp ${CMAKE_SOURCE_DIR}/include/clUtils/clFMatrix.hpp
)

target_link_libraries(
        Math_test PUBLIC
        Utils
        gtest_main
        BLAS::BLAS
        clUtils
        clblast
)

include(GoogleTest)
add_subdirectory(image)
add_subdirectory(controlSystem)
add_subdirectory(neuralNetwork)

add_executable(opencl_test opencl_test.cpp)
target_link_libraries(opencl_test PUBLIC OpenCL::OpenCL clblast)

gtest_discover_tests(Math_test)
if (COVERARE_ENABLED)
  setup_target_for_coverage_gcovr_html(NAME coverage
          EXECUTABLE ctest -j
          EXCLUDE "${CMAKE_SOURCE_DIR}/extern/*" "/usr/*" "${CMAKE_BINARY_DIR}/_deps/*" "${CMAKE_SOURCE_DIR}/test/*"
          "${CMAKE_SOURCE_DIR}/apps/*"
          )
endif ()
