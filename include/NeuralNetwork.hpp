/**
 * @brief This is a helper header file that includes all the headers related to neural networks
 */

#pragma once

#include "neuralNetwork/ActivationFunction.hpp"
#include "neuralNetwork/Model.hpp"
#include "neuralNetwork/Optimizer.hpp"

#include "neuralNetwork/Perceptron/MLPModel.hpp"
#include "neuralNetwork/Perceptron/MLPModelSerializer.hpp"

// Every available optimizations
#include "neuralNetwork/Perceptron/Optimization/DecayMomentumOptimization.hpp"
#include "neuralNetwork/Perceptron/Optimization/DecayOptimization.hpp"
#include "neuralNetwork/Perceptron/Optimization/MomentumOptimization.hpp"
#include "neuralNetwork/Perceptron/Optimization/RProPOptimization.hpp"
#include "neuralNetwork/Perceptron/Optimization/SGDOptimization.hpp"

#include "neuralNetwork/Perceptron/MLPOptimizer.hpp"
