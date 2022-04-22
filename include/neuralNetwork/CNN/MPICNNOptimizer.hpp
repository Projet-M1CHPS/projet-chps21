#pragma once

#include "CNNOptimizer.hpp"

class MPICNNOptimizer : public nnet::CNNOptimizer {
public:
  void reduceAll() override {
    mpi_reduce_cache_cnn();
    for () cache[0].reduce(caches);
  }

  void applyChanges() override {
    for (auto &cache : caches) { cache.apply(); }
    mlp_operation.updateModel();
  }

  void clearChanges() override;
};

std::unique_ptr<Optimizer::Operation> makeBatchOperation() override {
  return std::make_unique<MPIMLPOperation>();
}

private:
std::unique_ptr<nnet::MLPOptimizer> optimizer;
}
;
