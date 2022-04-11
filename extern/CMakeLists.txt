
include(FetchContent)

FetchContent_Declare(
        tscl
        GIT_REPOSITORY https://github.com/Thukisdo/tscl.git
        GIT_TAG c5689b42099b48c48dbfe4c80a7ead4014ac9822 # Fixed various warnings
)

FetchContent_MakeAvailable(tscl)
