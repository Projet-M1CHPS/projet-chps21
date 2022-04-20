set(MPI_INCLUDE_DIR ${INCLUDE_DIR}/mpiWrapper)
set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/controlSystem)

add_library(MPIControlSystem MPITrainingController.cpp ${MPI_INCLUDE_DIR}/MPITrainingController.hpp TrainingCollectionScatterer.cpp TrainingCollectionScatterer.hpp)

target_link_libraries(MPIControlSystem PUBLIC ControlSystem2 NeuralNetwork openclUtils OpenMP::OpenMP_CXX)
target_include_directories(MPIControlSystem PUBLIC ${CURRENT_INCLUDE_DIR} ${MPI_INCLUDE_DIR})