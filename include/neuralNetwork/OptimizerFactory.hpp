#pragma once
#include "ModelOptimizer.hpp"
#include "Perceptron/MLPOptimizer.hpp"
#include <memory>

namespace nnet {

  template<typename real = float, typename = std::enable_if<std::is_floating_point_v<real>>>
  class OptimizerFactory {
  public:
    template<typename... Args>
    static std::unique_ptr<ModelOptimizer<real>>
    makeMLPModelOptimizer(ModelOptimizerType type,
                          std::shared_ptr<OptimizationMethod<real>> opti_method) {
      std::unique_ptr<ModelOptimizer<real>> res = nullptr;

      switch (type) {
        case ModelOptimizerType::stochastic:
          res = std::make_unique<MLPModelStochOptimizer<real>>(opti_method);
          break;
        case ModelOptimizerType::batch:
          res = std::make_unique<MLPBatchOptimizer<real>>(opti_method);
          break;
        case ModelOptimizerType::minibatch:
          // FIXME: implement
          throw std::runtime_error("OptimizerFactory: minibatch not implemented yet");
        default:
          throw std::runtime_error("OptimizerFactory: unknown optimizer type");
      }

      return res;
    }

  private:
  };
}   // namespace nnet
