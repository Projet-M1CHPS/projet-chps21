#pragma once

#include "mpi.h"

#include "Perceptron/MLPOptimizer.hpp"

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
    nnet::MLPOptimizer::Operation *makeOperationImpl() override;
  };


  class MPIMLPOptimizer::Operation : public MLPOptimizer::Operation {
  public:
    explicit Operation(MPIMLPOptimizer &optimizer)
        : MLPOptimizer::Operation(optimizer), current_comm(MPI_COMM_WORLD) {}

    void setCommunicator(MPI_Comm comm) { this->current_comm = comm; }

    [[nodiscard]] MPI_Comm getCommunicator() const;

  protected:
    void reduceAll(cl::CommandQueue &queue) override;
    void applyChanges(cl::CommandQueue &queue) override;
    void clearChanges(cl::CommandQueue &queue) override;

  private:
    MPI_Comm current_comm{};
  };

}   // namespace nnet