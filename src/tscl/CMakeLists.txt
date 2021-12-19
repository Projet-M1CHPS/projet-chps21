
set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/tscl)
add_library(tscl
        logger.cpp ${CURRENT_INCLUDE_DIR}/logger.hpp
        time.cpp ${CURRENT_INCLUDE_DIR}/time.hpp
        version.cpp ${CURRENT_INCLUDE_DIR}/version.hpp
        ${INCLUDE_DIR}/tscl.hpp)
target_include_directories(tscl PUBLIC ${CURRENT_INCLUDE_DIR} ${INCLUDE_DIR})