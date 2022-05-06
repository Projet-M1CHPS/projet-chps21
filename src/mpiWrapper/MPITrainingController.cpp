#include "MPITrainingController.hpp"

#include "ParallelScheduler.hpp"
#include <chrono>

namespace chrono = std::chrono;
using namespace nnet;

namespace control {


  MPITrainingController::MPITrainingController(size_t maxEpoch, ModelEvolutionTracker &evaluator,
                                               nnet::OptimizationScheduler &scheduler)
      : TrainingController(maxEpoch, evaluator, scheduler) {}

  ControllerResult MPITrainingController::run() {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (size_t curr_epoch = 0; curr_epoch < max_epoch; curr_epoch++) {
      auto start = chrono::steady_clock::now();
      scheduler->run();
      auto end = chrono::steady_clock::now();
      auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

      if (rank == 0) {
        auto evaluation = evaluator->evaluate();
        if (is_verbose) {
          std::stringstream ss;
          ss << "(" << duration.count() << "ms) Epoch " << curr_epoch << ": " << evaluation
             << std::endl;
          tscl::logger(ss.str(), tscl::Log::Information);
        }
      }
    }

    return {0, "Training completed"};
  }
}   // namespace control