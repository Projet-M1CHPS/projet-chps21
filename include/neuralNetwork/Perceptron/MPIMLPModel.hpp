#pragma once

#include "MLPModel.hpp"
#include "MLPerceptron.hpp"
#include <mpi.h>

namespace nnet {

  /**
   * @brief A model for a multi-layer perceptron.
   */
  class MPIMLPModel : public MLPModel {
  public:
    using MLPModel::MLPModel;


    /**
     * @note The weights & biases are synchronized across all MPI processes
     * @brief Creates a random Model
     * @param seed The seed to be used for the randomization
     * @param topology The topology to be used for the model
     * @return a random model
     */
    static std::unique_ptr<MPIMLPModel>
    random(MLPTopology const &topology,
           af::ActivationFunctionType af = af::ActivationFunctionType::sigmoid);

    /**
     * @note The weights & biases are synchronized across all MPI processes
     * @brief Creates a random model that uses the relu/sigmoid activation functions.
     * The sigmoids are used for clipping the gradient to prevent its explosion
     * TODO: In the long run, this should be replaced by a L2 normalization
     * @param seed The seed to be used for the randomization
     * @param topology The topology to be used for the model
     * @return
     */
    static std::unique_ptr<MPIMLPModel> randomReluSigmoid(MLPTopology const &topology);

  private:
    std::unique_ptr<MLPerceptron> perceptron;
  };
}   // namespace nnet