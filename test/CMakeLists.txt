
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/e2239ee6043f73722e7aa812a459f54a28552929.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)
add_subdirectory(math)
add_subdirectory(image)
add_subdirectory(neuralNetwork)

if (COVERARE_ENABLED)
    setup_target_for_coverage_gcovr_html(NAME coverage
            EXECUTABLE ctest -j
            # Ensure every file is compiled to generate gcov files
            EXCLUDE "${CMAKE_SOURCE_DIR}/extern/*" "${CMAKE_BINARY_DIR}/_deps/*"
            "${CMAKE_SOURCE_DIR}/test/*"
            )
endif ()
