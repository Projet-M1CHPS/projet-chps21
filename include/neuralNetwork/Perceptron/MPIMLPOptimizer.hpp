#pragma once

#include "mpi.h"

#include "MLPOptimizer.hpp"

namespace nnet {

  class MPIMLPOptimizer : public MLPOptimizer {
  public:
    class Operation;
    using MLPOptimizer::MLPOptimizer;

    template<class optim, typename... Args>
    static std::unique_ptr<MPIMLPOptimizer> make(MLPModel &model, Args &&...args) {
      return std::make_unique<MPIMLPOptimizer>(
              model, std::make_unique<optim>(model.getPerceptron(), std::forward<Args>(args)...));
    }

    // Todo: override MLPOptimizer::makeMLPOperation()
    // std::unique_ptr<Operation> makeMLPOperation() override;

  private:
    std::unique_ptr<Optimizer::Operation> makeOperationImpl() override;
  };


  class MPIMLPOptimizer::Operation : public MLPOptimizer::Operation {
  public:
    explicit Operation(MPIMLPOptimizer &optimizer) : MLPOptimizer::Operation(optimizer) {}


  protected:
    void reduceAll(cl::CommandQueue &queue) override;
    void applyChanges(cl::CommandQueue &queue) override;
    void clearChanges(cl::CommandQueue &queue) override;
  };

}   // namespace nnet