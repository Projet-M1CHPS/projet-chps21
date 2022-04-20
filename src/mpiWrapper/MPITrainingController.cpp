#include "MPITrainingController.hpp"

#include "ParallelScheduler.hpp"
#include <chrono>

namespace chrono = std::chrono;
using namespace nnet;

// Todo: Remove after testing
static int win_mutex = -1;

// MPI variables
static int rank = -1;
static int nprocess = 0;

namespace control {

  namespace {

    // Printf overload
    void mpi_put(const std::string &message, tscl::Log::log_level level = tscl::Log::Debug) {
      std::string msg = message;
      msg.insert(0, "[P" + std::to_string(rank) + "]: ");
      msg.insert(0, rank == 0 ? "\033[0;33m" : "\033[0;35m");
      msg.append("\033[0m");

      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win_mutex);   // Lock the mutex
      tscl::logger(msg, level);
      MPI_Win_unlock(0, win_mutex);   // Unlock the mutex
    }

    // Create a single mutex for all the processes, accessible by MPI_Win_lock
    void create_output_mutex() {
      if (win_mutex)
        MPI_Win_create(&rank, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_mutex);
      MPI_Win_fence(0, win_mutex);
      mpi_put("Creating mutex");
    }

    bool areTensorEqual(const math::clFTensor &tensor1, const math::clFTensor &tensor2) {
      if (tensor1.getDepth() != tensor2.getDepth() || tensor1.getRows() != tensor2.getRows() ||
          tensor1.getCols() != tensor2.getCols()) {
        tscl::logger("Tensors are not same dimensions", tscl::Log::Error);
        return false;
      }
      auto tensor2_matrices = tensor2.getMatrices();
      int index = 0;
      return std::all_of(tensor1.getMatrices().cbegin(), tensor1.getMatrices().cend(),
                         [&index, &tensor2_matrices](const math::clFMatrix &matrix) {
                           auto f32_matrix1 = matrix.toFloatMatrix();
                           auto f32_matrix2 = tensor2_matrices.at(index).toFloatMatrix();
                           index++;
                           auto ret = std::equal(f32_matrix1.cbegin(), f32_matrix1.cend(),
                                                 f32_matrix2.cbegin(), f32_matrix2.cend());
                           return ret;
                         });
    }

    void initializeMPI(int &rank, int &world_size) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      create_output_mutex();
    }
  }   // namespace

  MPITrainingController::MPITrainingController(size_t maxEpoch, ModelEvolutionTracker &evaluator,
                                               nnet::OptimizationScheduler &scheduler)
      : TrainingController(maxEpoch, evaluator, scheduler) {}

  ControllerResult MPITrainingController::run() {
    initializeMPI(rank, nprocess);
    mpi_put("Running MPI training controller...");

    for (size_t curr_epoch = 0; curr_epoch < max_epoch; curr_epoch++) {
      auto start = chrono::steady_clock::now();
      scheduler->run();
      auto end = chrono::steady_clock::now();
      auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
      // Todo: Gather the results in the master process
      // Todo: Evaluate the results in the master process
      if (false) {
        auto evaluation = evaluator->evaluate();
        if (is_verbose) {
          std::stringstream ss;
          ss << "(" << duration.count() << "ms) Epoch " << curr_epoch << ": " << evaluation
             << std::endl;
          tscl::logger(ss.str(), tscl::Log::Information);
        }
      } else {
        if (is_verbose) {
          std::stringstream ss;
          ss << "(" << duration.count() << "ms) Epoch " << curr_epoch << std::endl;
          tscl::logger(ss.str(), tscl::Log::Information);
        }
      }
      // Todo: Scatter the evaluation results to all processes
    }

    return {0, "Training completed"};
  }
}   // namespace control