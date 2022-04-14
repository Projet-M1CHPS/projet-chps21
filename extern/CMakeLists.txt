
include(FetchContent)

FetchContent_Declare(
        tscl
        GIT_REPOSITORY https://github.com/Thukisdo/tscl.git
        GIT_TAG 961ea9a6ea5867d605a0209592ca3f22bbb3c279 # Fixed various warnings
)

FetchContent_MakeAvailable(tscl)
