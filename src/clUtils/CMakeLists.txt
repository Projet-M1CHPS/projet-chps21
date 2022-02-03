
set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/clUtils)
add_library(clUtils STATIC clWrapper.cpp ${CURRENT_INCLUDE_DIR}/clWrapper.hpp)
target_include_directories(clUtils PUBLIC ${CURRENT_INCLUDE_DIR})
target_link_libraries(clUtils PUBLIC OpenCL::OpenCL tscl::tscl)