
set(CURRENT_INCLUDE_DIR ${INCLUDE_DIR}/neuralNetwork/Perceptron)
add_library(Perceptron STATIC
        MLPerceptron.cpp ${CURRENT_INCLUDE_DIR}/MLPerceptron.hpp
        MLPSerializer.cpp ${CURRENT_INCLUDE_DIR}/MLPSerializer.hpp

        MLPModel.cpp ${CURRENT_INCLUDE_DIR}/MLPModel.hpp
        MLPModelSerializer.cpp ${CURRENT_INCLUDE_DIR}/MLPModelSerializer.hpp

        ${CURRENT_INCLUDE_DIR}/Optimization/Optimization.hpp

        # Every optimization
        Optimization/SGDOptimization.cpp ${CURRENT_INCLUDE_DIR}/Optimization/SGDOptimization.hpp
        Optimization/DecayOptimization.cpp ${CURRENT_INCLUDE_DIR}/Optimization/DecayOptimization.hpp
        Optimization/MomentumOptimization.cpp ${CURRENT_INCLUDE_DIR}/Optimization/MomentumOptimization.hpp
        Optimization/DecayMomentumOptimization.cpp ${CURRENT_INCLUDE_DIR}/Optimization/DecayMomentumOptimization.hpp

        # Optimizers
        ${CURRENT_INCLUDE_DIR}/MLPOptimizer.hpp
        MLPOptimizer.cpp ${CURRENT_INCLUDE_DIR}/MLPOptimizer.hpp
        ActivationFunction.cpp ${CURRENT_INCLUDE_DIR}/ActivationFunction.hpp
        )

target_include_directories(Perceptron
        PUBLIC ${INCLUDE_DIR} ${CURRENT_INCLUDE_DIR} ${INCLUDE_DIR}/neuralNetwork ${CURRENT_INCLUDE_DIR}/Optimization)

target_link_libraries(Perceptron PUBLIC tscl::tscl openclUtils Math)

if (USE_MPI)
    message(STATUS "Using MPI for MLP")
    add_library(MPIPerceptron STATIC
            MPIMLPOptimizer.cpp ${CURRENT_INCLUDE_DIR}/MPIMLPOptimizer.hpp
            MPIMLPModel.cpp ${CURRENT_INCLUDE_DIR}/MPIMLPModel.hpp
            )
    target_include_directories(MPIPerceptron
            PUBLIC ${INCLUDE_DIR} ${CURRENT_INCLUDE_DIR} ${INCLUDE_DIR}/neuralNetwork ${CURRENT_INCLUDE_DIR}/Optimization)
    target_link_libraries(MPIPerceptron PUBLIC tscl::tscl openclUtils Math Perceptron)
endif ()
