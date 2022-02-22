#pragma once
#include "MLPOptimizer.hpp"

namespace nnet {

  class MLPBatchOptimizer : public MLPOptimizer {
  public:
    explicit MLPBatchOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm);

    void optimize(const std::vector<math::FloatMatrix> &inputs,
                  const std::vector<math::FloatMatrix> &targets) override;

    /**
     * @brief Builds a new optimization algorithm, forwarding parameters, and returns an
     * optimizer build around it
     * @tparam Optim The optimization to use
     * @tparam Args The arguments to pass to the optimization
     * @param model The model to optimize
     * @param args The arguments to pass to the optimization
     * @return A new optimizer build around the new optimization algorithm
     */
    template<class Optim, typename... Args, typename = std::is_base_of<nnet::Optimization, Optim>>
    static std::unique_ptr<MLPBatchOptimizer> make(MLPModel &model, Args &&...args) {
      return std::make_unique<MLPBatchOptimizer>(
              model, std::make_unique<Optim>(model.getPerceptron(), std::forward<Args>(args)...));
    }


  protected:
    void forward(math::FloatMatrix const &inputs);

    void computeGradient();

    //
    std::vector<math::FloatMatrix> layers, layers_af;

    //
    BackpropStorage storage;

    std::vector<math::FloatMatrix> avg_errors;
    std::vector<math::FloatMatrix> avg_gradients;
  };

}   // namespace nnet