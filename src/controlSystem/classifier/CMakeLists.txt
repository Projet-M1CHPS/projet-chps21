

add_library(Classifier
        classifierCollection.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/classifier/classifierCollection.hpp
        classifierController.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/classifier/classifierController.hpp
        classifierInputSet.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/classifier/classifierInputSet.hpp
        classifierTracker.cpp ${CMAKE_SOURCE_DIR}/include/controlSystem/classifier/classifierTracker.hpp
        )

target_include_directories(Classifier PUBLIC ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/controlSystem/classifier
        )