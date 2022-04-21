add_executable(
        Math_test
        Math_test.cpp
        clFMatrix_test.cpp
        clFTensor_test.cpp)

target_link_libraries(
        Math_test PUBLIC
        Utils
        ${BLAS_LIBRARIES}
        Math
        gtest
        openclUtils
        clblast
)

gtest_discover_tests(Math_test)
