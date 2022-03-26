
find_package(Curses REQUIRED)

set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/clUtils)
add_library(clUtils STATIC
        clWrapper.cpp ${CURRENT_INCLUDE_DIR}/clWrapper.hpp
        clFMatrix.cpp ${CURRENT_INCLUDE_DIR}/clFMatrix.hpp
        clFTensor.cpp ${CURRENT_INCLUDE_DIR}/clFTensor.hpp
        clQueueHandler.cpp ${CURRENT_INCLUDE_DIR}/clQueueHandler.hpp
        clKernelMap.cpp ${CURRENT_INCLUDE_DIR}/clKernelMap.hpp
        clPlatformSelector.cpp ${CURRENT_INCLUDE_DIR}/clPlatformSelector.hpp
        )
target_include_directories(clUtils PUBLIC ${CURRENT_INCLUDE_DIR})
target_link_libraries(clUtils PUBLIC OpenCL::OpenCL tscl::tscl Math clblast ncurses)