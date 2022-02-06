
add_executable(ControlSystem_test
        ControllerParametersTest.cpp
        classifierClassTest.cpp
        classifierInputSetTest.cpp
        #classifierCollectionTest.cpp
        #classifierControllerTest.cpp
        #classifierTrackerTest.cpp
        )

target_include_directories(ControlSystem_test
        PRIVATE
        ${INCLUDE_DIR}
        )

target_link_libraries(
        ControlSystem_test PUBLIC
        ControlSystem
        Classifier
        gtest_main
)
gtest_discover_tests(ControlSystem_test)

