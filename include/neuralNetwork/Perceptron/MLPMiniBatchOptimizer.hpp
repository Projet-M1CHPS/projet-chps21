#pragma once
#include "MLPBatchOptimizer.hpp"


namespace nnet {

  class MLPMiniBatchOptimizer : public MLPBatchOptimizer {
  public:
    explicit MLPMiniBatchOptimizer(MLPModel &model, std::shared_ptr<Optimization> tm,
                                   size_t batch_size = 8);

    void optimize(const std::vector<math::clFMatrix> &inputs,
                  const std::vector<math::clFMatrix> &targets) override;

    /**
     * @brief Builds a new optimization algorithm, forwarding parameters, and returns an
     * optimizer build around it
     * @tparam Optim The optimization algorithm to use
     * @tparam Args Args to forward to the optimization algorithm
     * @param model The model to optimize
     * @param batch_size The size of the mini-baches
     * @param args Args of the optimizer
     * @return A new optimizer and allocated dedicated optimization algorithm
     */
    template<class Optim, typename... Args, typename = std::is_base_of<nnet::Optimization, Optim>>
    static std::unique_ptr<MLPMiniBatchOptimizer> make(MLPModel &model, size_t batch_size,
                                                       Args &&...args) {
      return std::make_unique<MLPMiniBatchOptimizer>(
              model,
              std::make_unique<Optim>(model.getPerceptron(), model.getClWrapper(),
                                      std::forward<Args>(args)...),
              batch_size);
    }

  private:
    size_t batch_size;
  };

}   // namespace nnet