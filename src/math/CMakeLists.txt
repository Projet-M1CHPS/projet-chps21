
set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/math)
add_library(Math STATIC
        clFMatrix.cpp ${CURRENT_INCLUDE_DIR}/clFMatrix.hpp
        clFTensor.cpp ${CURRENT_INCLUDE_DIR}/clFTensor.hpp
        ${CURRENT_INCLUDE_DIR}/Matrix.hpp
        )
target_include_directories(Math PUBLIC ${CURRENT_INCLUDE_DIR})
target_link_libraries(Math PUBLIC OpenCL::OpenCL openclUtils)