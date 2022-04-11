
find_package(Curses REQUIRED)

set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/openclUtils)
add_library(openclUtils STATIC
        clWrapper.cpp ${CURRENT_INCLUDE_DIR}/clWrapper.hpp
        clKernelMap.cpp ${CURRENT_INCLUDE_DIR}/clKernelMap.hpp
        clPlatformSelector.cpp ${CURRENT_INCLUDE_DIR}/clPlatformSelector.hpp
        )
target_include_directories(openclUtils PUBLIC ${CURRENT_INCLUDE_DIR})
target_link_libraries(openclUtils PUBLIC OpenCL::OpenCL tscl::tscl clblast ncurses Boost::boost Boost::filesystem)