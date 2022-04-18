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

    void sendTensor(const InputSet &input_set, const math::clFTensor &tensor, int dest) {
      mpi_put("> sendTensor(...)");

      const auto &tensor_buffer = tensor.getBuffer();
      MPI_Request dimension_request;

      // Get tensor dimensions
      size_t depth = tensor.getDepth();
      size_t rows = tensor.getRows();
      size_t cols = tensor.getCols();
      // Create an aligned tuple of dimensions
      std::tuple<unsigned long, unsigned long, unsigned long, unsigned long> dims =
              std::make_tuple(tensor.getOffset(), rows, cols, depth);

      // Send the dimensions [OFFSET, ROWS, COLS, DEPTH]
      mpi_put("Sending tensor dimensions (" + std::to_string(tensor.getOffset()) + "," +
              std::to_string(rows) + "," + std::to_string(cols) + "," + std::to_string(depth) +
              ") to " + std::to_string(dest));
      MPI_Isend(&dims, 4, MPI_UNSIGNED_LONG, dest, 0, MPI_COMM_WORLD, &dimension_request);

      // Wait for the dimensions to be sent
      MPI_Wait(&dimension_request, MPI_STATUS_IGNORE);
      mpi_put("Sent tensor dimensions to " + std::to_string(dest));

      // Send the tensor
      mpi_put("Sending tensor to " + std::to_string(dest));
      for (size_t z = 0; z < tensor.getMatrices().size(); z++) {
        auto matrix = tensor[z];
        MPI_Send(matrix.toFloatMatrix().getData(), (int) matrix.size(), MPI_FLOAT, dest, 0,
                 MPI_COMM_WORLD);
      }
      mpi_put("Sent tensor " + std::to_string(dest) + " to process " + std::to_string(dest));

      // Send samples ids
      auto class_ids = input_set.getSamplesIds();
      mpi_put("Sending samples ids to " + std::to_string(dest));
      MPI_Send(class_ids.data(), (int) class_ids.size(), MPI_UNSIGNED_LONG, dest, 0,
               MPI_COMM_WORLD);
      mpi_put("Sent class ids to " + std::to_string(dest));

      // Send samples class ids
      auto samples_class_ids = input_set.getSamplesClassIds();
      mpi_put("Sending samples class ids to " + std::to_string(dest));
      MPI_Send(samples_class_ids.data(), (int) samples_class_ids.size(), MPI_LONG, dest, 0,
               MPI_COMM_WORLD);
      mpi_put("Sent samples class ids to " + std::to_string(dest));

      mpi_put("< sendTensor(...)");
    }

    void scatterTrainingSet(const InputSet &input_set, InputSet &training_set) {
      mpi_put("> scatterTrainingSet(...)");

      const auto &tensors = input_set.getTensors();
      auto class_ids = input_set.getSamplesIds();

      mpi_put("Training_set contains " + std::to_string(input_set.getTensorCount()) + " tensors");

      size_t tensor_part = input_set.getTensorCount() / nprocess;
      size_t tensor_remainder = input_set.getTensorCount() % nprocess;

      std::vector<unsigned long> tensor_counts(nprocess);
      for (size_t p = 0; p < nprocess; p++) {
        tensor_counts[p] = tensor_part + (p < tensor_remainder-- ? 1 : 0);
        mpi_put("Sending " + std::to_string(tensor_counts.at(p)) + " tensors to process " +
                std::to_string(p));
      }

      unsigned long unused;
      // Todo: Scatter tensor_counts
      MPI_Scatter(tensor_counts.data(), 1, MPI_UNSIGNED_LONG, &unused, 1, MPI_UNSIGNED_LONG, 0,
                  MPI_COMM_WORLD);

      for (size_t t = 0; t < tensor_counts.at(0); t++) {
        // Local tensor mapping
      }

      size_t current_tensor_idx = 0;
      for (size_t p = 1; p < nprocess; p++)
        for (size_t j = 0; j < tensor_counts.at(p); j++)
          sendTensor(input_set, tensors.at(current_tensor_idx++), p);

      mpi_put("< scatterTrainingSet(...)");
    }

    void receiveTensor(math::clFTensor &tensor, std::vector<size_t> &ids,
                       std::vector<long> class_ids, int tensor_index) {
      mpi_put("> receiveTensor(...)");
      MPI_Status status;

      // Receive the tensors dimensions [OFFSET, ROWS, COLS, DEPTH]
      std::vector<unsigned long> tensor_dimensions(4);
      MPI_Recv(tensor_dimensions.data(), 4, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      mpi_put("Received tensor " + std::to_string(tensor_index) + " dimensions: (" +
              std::to_string(tensor_dimensions.at(0)) + ", " +
              std::to_string(tensor_dimensions.at(1)) + ", " +
              std::to_string(tensor_dimensions.at(2)) + ", " +
              std::to_string(tensor_dimensions.at(3)) + ")");

      // Receive the tensor
      math::clFTensor tmp_tensor(tensor_dimensions.at(1), tensor_dimensions.at(2),
                                 tensor_dimensions.at(3));
      math::FloatMatrix tmp_matrix(tensor_dimensions.at(1), tensor_dimensions.at(2));
      for (size_t z = 0; z < tensor_dimensions.at(3); z++) {
        MPI_Recv(tmp_matrix.getData(), (int) (tensor_dimensions.at(1) * tensor_dimensions.at(2)),
                 MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        mpi_put("Received fMatrix " + std::to_string(z));
        tmp_tensor[z] = math::clFMatrix(tmp_matrix);
      }
      mpi_put("Received entire tensor " + std::to_string(tensor_index));

      // Receive the samples ids
      int nb_ids = 0;
      MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_UNSIGNED_LONG, &nb_ids);
      ids.resize(nb_ids);
      MPI_Recv(ids.data(), nb_ids, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      mpi_put("Received " + std::to_string(nb_ids) + " ids");

      // Receive the samples class ids
      int nb_class_ids = 0;
      MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_UNSIGNED_LONG, &nb_class_ids);
      class_ids.resize(nb_class_ids);
      MPI_Recv(class_ids.data(), nb_class_ids, MPI_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      mpi_put("Received " + std::to_string(nb_class_ids) + " class ids");

      // Todo: assign referenced tensor


      mpi_put("< receiveTensor(...)");
    }

    void receiveTrainingSet(InputSet &training_set) {
      mpi_put("> receiveTrainingSet(...)");
      unsigned long tensor_count = 0;

      // Receive the number of tensors to be received
      MPI_Scatter(nullptr, 1, MPI_UNSIGNED_LONG, &tensor_count, 1, MPI_UNSIGNED_LONG, 0,
                  MPI_COMM_WORLD);
      mpi_put("Received " + std::to_string(tensor_count) + " tensor_count");

      for (int t = 0; t < tensor_count; t++) {
        math::clFTensor tensor;
        std::vector<size_t> ids;
        std::vector<long> class_ids;
        receiveTensor(tensor, ids, class_ids, t);

        // Add the tensor to the training set
        // training_set.append(tensor, ids, class_ids);
      }

      mpi_put("< receiveTrainingSet(...)");
    }
  }   // namespace

  MPITrainingController::MPITrainingController(size_t maxEpoch, ModelEvolutionTracker &evaluator,
                                               nnet::OptimizationScheduler &scheduler)
      : TrainingController(maxEpoch, evaluator, scheduler) {}

  ControllerResult MPITrainingController::run() {
    initializeMPI(rank, nprocess);
    mpi_put("Running MPI training controller...");


    return {0, "Training halted"};

    for (size_t curr_epoch = 0; curr_epoch < max_epoch; curr_epoch++) {
      auto start = chrono::steady_clock::now();
      scheduler->run();
      auto end = chrono::steady_clock::now();
      auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
      // Todo: Gather the results in the master process
      // Todo: Evaluate the results in the master process
      auto evaluation = evaluator->evaluate();
      if (is_verbose) {
        std::stringstream ss;
        ss << "(" << duration.count() << "ms) Epoch " << curr_epoch << ": " << evaluation
           << std::endl;
        tscl::logger(ss.str(), tscl::Log::Information);
      }
      // Todo: Scatter the evaluation results to all processes
    }

    return {0, "Training completed"};
  }
}   // namespace control