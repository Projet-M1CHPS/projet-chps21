set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/neuralNetwork/OptimizationScheduler)


add_library(OptimizationScheduler STATIC
        ${CURRENT_INCLUDE_DIR}/OptimizationScheduler.hpp
        BatchProgression.cpp ${CURRENT_INCLUDE_DIR}/BatchProgression.hpp
        BatchOptimizationScheduler.cpp ${CURRENT_INCLUDE_DIR}/BatchOptimizationScheduler.hpp
        SchedulerProfiler.cpp ${CURRENT_INCLUDE_DIR}/SchedulerProfiler.hpp
        ParallelScheduler.cpp ${CURRENT_INCLUDE_DIR}/ParallelScheduler.hpp
        )
target_include_directories(OptimizationScheduler PUBLIC ${CURRENT_INCLUDE_DIR})
target_link_libraries(OptimizationScheduler PUBLIC Math nnet)


if (USE_MPI)
    add_library(MPIOptimizationScheduler STATIC
            MPIParallelScheduler.cpp ${CURRENT_INCLUDE_DIR}/MPIParallelScheduler.hpp
            ../Perceptron/MPIMLPOptimizer.cpp ${INCLUDE_DIR}/neuralNetwork/Perceptron/MPIMLPOptimizer.hpp
            )
    target_include_directories(MPIOptimizationScheduler PUBLIC ${CURRENT_INCLUDE_DIR})
    target_link_libraries(MPIOptimizationScheduler PUBLIC Math nnet OptimizationScheduler MPIPerceptron)
endif ()