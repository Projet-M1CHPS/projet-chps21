#pragma once
#include "MLPOptimizer.hpp"

namespace nnet {


  /**
   * @brief Stochastic optimizer, training the model using one input at a time
   */
  class MLPStochOptimizer final : public MLPOptimizer {
  public:
    MLPStochOptimizer(MLPModel &model, std::unique_ptr<Optimization> tm);
    MLPStochOptimizer(MLPerceptron &perceptron, std::unique_ptr<Optimization> tm);

    math::clFMatrix optimize(const math::clFMatrix &input, const math::clFMatrix &target);

    void optimize(const std::vector<math::clFTensor> &inputs,
                  const std::vector<math::clFTensor> &targets) override;

    /**
     * @brief Builds a new optimization algorithm, forwarding parameters, and returns an
     * optimizer build around it
     * @tparam Optim The type of optimization algorithm to use
     * @tparam Args The type of arguments to pass to the optimization algorithm
     * @param model The model to optimize
     * @param args The arguments to pass to the optimization algorithm
     * @return A new optimizer build around the new optimization algorithm
     */
    template<class Optim, typename... Args, typename = std::is_base_of<nnet::Optimization, Optim>>
    static std::unique_ptr<MLPStochOptimizer> make(MLPModel &model, Args &&...args) {
      return std::make_unique<MLPStochOptimizer>(
              model, std::make_unique<Optim>(model.getPerceptron(), std::forward<Args>(args)...));
    }

  private:
    void forward(math::clFMatrix const &inputs, cl::CommandQueue &queue);
    void backward(math::clFMatrix const &target, cl::CommandQueue &queue);

  private:
    //
    std::vector<math::clFMatrix> layers, layers_af;

    //
    BackpropStorage storage;
  };

}   // namespace nnet