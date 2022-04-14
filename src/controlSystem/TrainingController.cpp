#include "TrainingController.hpp"

#include "ParallelScheduler.hpp"
#include <chrono>

namespace chrono = std::chrono;
using namespace nnet;

namespace control {

  TrainingController::TrainingController(size_t max_epoch, ModelEvolutionTracker &evaluator,
                                         nnet::OptimizationScheduler &scheduler)
      : max_epoch(max_epoch), evaluator(&evaluator), scheduler(&scheduler) {}

  ControllerResult TrainingController::run() {
    for (size_t curr_epoch = 0; curr_epoch < max_epoch; curr_epoch++) {
      auto start = chrono::steady_clock::now();
      scheduler->run();
      auto end = chrono::steady_clock::now();
      auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
      auto evaluation = evaluator->evaluate();
      if (is_verbose)
        std::cout << "(" << duration.count() << std::endl<< "ms) Epoch " << curr_epoch << ": " << evaluation
                  << std::endl;
    }

    return {0, "Training completed"};
  }
}   // namespace control