
include(FetchContent)

FetchContent_Declare(
        tscl
        GIT_REPOSITORY https://github.com/Thukisdo/tscl.git
        GIT_TAG 51edc0c229e957e125c8ef8e58b1891d165cd2aa # pre-release-0.1.0.alpha
)

FetchContent_MakeAvailable(tscl)