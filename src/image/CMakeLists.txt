
set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/image)
add_library(Image STATIC
        Image.cpp ${CURRENT_INCLUDE_DIR}/Image.hpp
        TransformEngine.cpp
        Transform.cpp ${CURRENT_INCLUDE_DIR}/Transform.hpp
        ${CMAKE_SOURCE_DIR}/extern/stb_image.h
        ${CMAKE_SOURCE_DIR}/extern/stb_image_write.h
        )
set_target_properties(Image PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Uses the stb library located in dependency
target_include_directories(Image PUBLIC "${CURRENT_INCLUDE_DIR}" "${INCLUDE_DIR}" "${CMAKE_SOURCE_DIR}/extern")
target_link_libraries(Image PUBLIC clUtils Math)


