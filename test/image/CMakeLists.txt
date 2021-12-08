

add_executable(Image_test Image_test.cpp Transform_test.cpp)

target_link_libraries(
        Image_test PUBLIC
        Image
        gtest_main
)

target_code_coverage(Image_test ALL)

gtest_discover_tests(Image_test)