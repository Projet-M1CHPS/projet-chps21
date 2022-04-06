add_executable(
        Math_test
        Math_test.cpp
        ${CMAKE_SOURCE_DIR}/include/Matrix.hpp
        clFMatrix_test.cpp ${CMAKE_SOURCE_DIR}/include/clUtils/clFMatrix.hpp
        clFTensor_test.cpp)

target_link_libraries(
        Math_test PUBLIC
        Utils
        BLAS::BLAS
        gtest
        clUtils
        clblast
)

gtest_discover_tests(Math_test)
