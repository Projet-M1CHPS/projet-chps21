

# Adapted from the following tutorial:
# https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/
find_package(Doxygen)

# Dont force the use of Doxygen, it may not be installed
if (Doxygen_FOUND)
  set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs)
  set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/html/index.html)
  set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
  set(DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  configure_file(${DOXYFILE_IN} ${DOXYFILE})

  add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
          COMMAND Doxygen::doxygen ${DOXYFILE}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          MAIN_DEPENDENCY ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
          COMMENT "Generating docs")

  add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

else ()
  message(WARNING "Doxygen not found, documentation generation disabled")
endif ()