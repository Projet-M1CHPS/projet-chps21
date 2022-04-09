add_executable(
        Math_test
        Math_test.cpp
        clFMatrix_test.cpp
        clFTensor_test.cpp)

target_link_libraries(
        Math_test PUBLIC
        Utils
        BLAS::BLAS
        Math
        gtest
        clUtils
        clblast
)

gtest_discover_tests(Math_test)
