
add_executable(ControlSystem_test InputSet_test.cpp InputSetLoader_test.cpp)
target_include_directories(ControlSystem_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(
        ControlSystem_test PUBLIC
        controlSystem
        gtest_main
)

target_code_coverage(ControlSystem_test ALL)
gtest_discover_tests(ControlSystem_test)