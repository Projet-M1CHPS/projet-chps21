/**
 * @brief This is a helper header file that includes all the headers related to neural networks
 */
#pragma once


#include "neuralNetwork/Model.hpp"
#include "neuralNetwork/Optimizer.hpp"

#include "neuralNetwork/Perceptron/MLPModelSerializer.hpp"
#include "neuralNetwork/Perceptron/MPIMLPModel.hpp"

#include "neuralNetwork/Perceptron/ActivationFunction.hpp"

// Every available optimizations
#include "neuralNetwork/Perceptron/Optimization/DecayMomentumOptimization.hpp"
#include "neuralNetwork/Perceptron/Optimization/DecayOptimization.hpp"
#include "neuralNetwork/Perceptron/Optimization/MomentumOptimization.hpp"
#include "neuralNetwork/Perceptron/Optimization/SGDOptimization.hpp"

// Optimizer
#include "neuralNetwork/Perceptron/MPIMLPOptimizer.hpp"
