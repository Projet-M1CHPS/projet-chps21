
add_executable(InputSet_test InputSet_test.cpp)
target_include_directories(InputSet_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(
        InputSet_test PUBLIC
        controlSystem
        gtest_main
)

target_code_coverage(InputSet_test ALL)
gtest_discover_tests(InputSet_test)