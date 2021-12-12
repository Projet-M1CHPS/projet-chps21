#pragma once

namespace nnet {


  template<typename real = float, typename = std::enable_if<std::is_floating_point_v<real>>>
  class OptimizerFactory {
  private:
  public:
  };
}   // namespace nnet
