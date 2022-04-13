#include "MPITrainingController.hpp"
#include <chrono>

namespace chrono = std::chrono;

namespace control {

  namespace {
    void mpi_put(std::string const &message, tscl::Log::log_level level = tscl::Log::Debug) {
      int mpi_rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
      std::stringstream ss;
      ss << "[P" << mpi_rank << "]: " << message;
      tscl::logger(ss.str(), level);
    }

    std::vector<math::clFTensor> setupTargets(const InputSet &input_set) {
      std::vector<math::clFTensor> res;
      size_t nclass = input_set.getClassIds().size();

      for (size_t sample_index = 0; auto &tensor : input_set.getTensors()) {
        size_t size = tensor.getZ();
        math::clFTensor buf(nclass, 1, size);

        for (size_t j = 0; j < size; j++) {
          math::FloatMatrix mat(nclass, 1);
          mat.fill(0.0f);
          auto class_id = input_set.getClassOf(sample_index);
          mat(class_id, 0) = 1.0f;
          buf.getMatrix(j) = mat;
          sample_index++;
        }
        res.push_back(buf);
      }
      return res;
    }

    chrono::milliseconds runEpoch(nnet::Optimizer &optimizer, const InputSet &input_set,
                                  const std::vector<math::clFTensor> &targets) {
      mpi_put("Running epoch...", tscl::Log::Trace);
      auto start = chrono::high_resolution_clock::now();
      optimizer.optimize(input_set.getTensors(), targets);
      auto end = chrono::high_resolution_clock::now();
      optimizer.update();
      return chrono::duration_cast<chrono::milliseconds>(end - start);
    }

    void initializeMPI(int &rank, int &world_size) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    void scatterTrainingSet(const InputSet &input_set, InputSet &training_set) {
      mpi_put("Scattering training set...", tscl::Log::Warning);
      int nprocess, rank;
      initializeMPI(rank, nprocess);
      assert(rank == 0);

      const auto &tensors = input_set.getTensors();
      auto class_ids = input_set.getClassIds();

      // region Assertions on targets
      mpi_put("Training_set tensor count: " + std::to_string(input_set.getTensorCount()),
              tscl::Log::Warning);

      cl::CommandQueue queue(utils::cl_wrapper.getContext(), utils::cl_wrapper.getDefaultDevice());

      size_t tensor_part = input_set.getTensorCount() / nprocess;
      size_t tensor_remainder = input_set.getTensorCount() % nprocess;

      size_t current_tensor_idx = 0;
      size_t current_class_idx = 0;
      for (int i = 1; i < nprocess; i++) {
        size_t send_count = tensor_part;
        if (i <= tensor_remainder) send_count++;

        mpi_put("Sending " + std::to_string(send_count) + " tensors to process " +
                        std::to_string(i),
                tscl::Log::Warning);
        // Send tensor count
        MPI_Send(&send_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        mpi_put("Sending tensors...", tscl::Log::Warning);

        for (size_t j = 0; j < send_count; j++) {
          const auto &tensor = input_set.getTensor(current_tensor_idx);
          const auto &buffer = tensor.getBuffer();

          // Send the data
          int size = (int) (tensor.getZ() * tensor.getX() * tensor.getY());
          float *map = (float *) queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_READ, 0,
                                                        size * sizeof(float));
          mpi_put(std::to_string(j) + " >> Sending tensor " + std::to_string(current_tensor_idx) +
                          "[" + std::to_string(size) + " floats] to process " + std::to_string(i),
                  tscl::Log::Warning);
          MPI_Send(map, size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
          mpi_put(std::to_string(j) + " >> Sent tensor " + std::to_string(current_tensor_idx) +
                          " to process " + std::to_string(i),
                  tscl::Log::Warning);
          queue.enqueueUnmapMemObject(buffer, map);

          // Send the dimensions
          std::vector<unsigned long> dimensions(3);
          dimensions[0] = tensor.getX();
          dimensions[1] = tensor.getY();
          dimensions[2] = tensor.getZ();
          mpi_put(std::to_string(j) + " >> Sending dimensions [" + std::to_string(dimensions[0]) +
                          ", " + std::to_string(dimensions[1]) + ", " +
                          std::to_string(dimensions[2]) + "] of tensor " +
                          std::to_string(current_tensor_idx) + " to process " + std::to_string(i),
                  tscl::Log::Warning);
          MPI_Send(dimensions.data(), 3, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
          mpi_put(std::to_string(j) + " >> Sent dimensions of tensor " +
                          std::to_string(current_tensor_idx) + " to process " + std::to_string(i),
                  tscl::Log::Warning);

          // Send the class ids
          mpi_put(std::to_string(j) + " >> Sending " + std::to_string(tensor.getZ()) +
                          " class ids of tensor " + std::to_string(current_tensor_idx) +
                          " to process " + std::to_string(i),
                  tscl::Log::Warning);
          auto begin_ptr = class_ids.at(current_class_idx);
          MPI_Send(&begin_ptr, (int) tensor.getZ(), MPI_LONG, i, 0, MPI_COMM_WORLD);
          mpi_put(std::to_string(j) + " >> Sent class ids of tensor " +
                          std::to_string(current_tensor_idx) + " to process " + std::to_string(i),
                  tscl::Log::Warning);

          current_class_idx += tensor.getZ();
          current_tensor_idx++;
        }
      }

      // Split my own input_set into training_set
      for (size_t i = current_tensor_idx; i < input_set.getTensorCount(); i++) {
        auto &tensor = input_set.getTensor(i);
        std::vector<size_t> sub_class_ids;
        std::vector<long> sub_class_ids_long;
        assert(tensor.getZ() + current_tensor_idx <= input_set.getClassIds().size());
        for (size_t j = current_class_idx; j < tensor.getZ() + current_class_idx; j++) {
          sub_class_ids.push_back((size_t) class_ids[j]);
          sub_class_ids_long.push_back(class_ids[j]);
        }

        if (sub_class_ids.size() != tensor.getZ())
          throw std::runtime_error(
                  "Class ids & tensor size do not match: " + std::to_string(sub_class_ids.size()) +
                  " vs " + std::to_string(tensor.getZ()));
        else
          mpi_put(std::to_string(i) + " >> Appending tensor");
        training_set.append(tensor.shallowCopy(), sub_class_ids, sub_class_ids_long);
      }
      // Reallocate the input_set
      queue.finish();
    }

    void receiveTrainingSet(InputSet &training_set) {
      mpi_put("Receiving training set...");
      MPI_Status status;
      std::vector<math::clFTensor> tensors;

      std::vector<float> training_set_raw;
      int tensor_count = 0, recv_size = 0;
      cl::CommandQueue queue(utils::cl_wrapper.getContext(), utils::cl_wrapper.getDefaultDevice());
      MPI_Recv(&tensor_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
      mpi_put("Receiving " + std::to_string(tensor_count) + " tensors");
      for (int i = 0; i < tensor_count; i++) {
        // Receive tensor size
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_FLOAT, &recv_size);
        mpi_put(std::to_string(i) + " >> Receiving " + std::to_string(recv_size) + " floats");
        training_set_raw.resize(recv_size);
        MPI_Recv(training_set_raw.data(), recv_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        mpi_put(std::to_string(i) + " >> Received " + std::to_string(recv_size) + " floats");

        // Receive dimensions
        std::vector<unsigned long> dimensions(3);
        MPI_Recv(dimensions.data(), 3, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        mpi_put(std::to_string(i) + " >> Received dimensions [" + std::to_string(dimensions[0]) +
                ", " + std::to_string(dimensions[1]) + ", " + std::to_string(dimensions[2]) + "]");

        // Initialize tensor
        const auto &tensor = math::clFTensor(dimensions[0], dimensions[1], dimensions[2]);
        mpi_put(std::to_string(i) + " >> Initializing tensor");
        tensors.emplace_back(tensor);
        queue.enqueueWriteBuffer(tensor.getBuffer(), CL_TRUE, 0, recv_size * sizeof(float),
                                 (void *) training_set_raw.data());
        mpi_put(std::to_string(i) + " >> Mapping tensor");

        // Receive class ids
        int class_id_count = 0;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_LONG, &class_id_count);
        mpi_put(std::to_string(i) + " >> Receiving " + std::to_string(class_id_count) +
                " class ids");
        std::vector<size_t> class_ids(class_id_count);
        MPI_Recv((void *) class_ids.data(), class_id_count, MPI_LONG, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        mpi_put(std::to_string(i) + " >> Received " + std::to_string(class_id_count) +
                " class ids");

        // Append tensor
        // Todo: ids
        std::vector<long> copy_class_ids;
        for (auto id : class_ids) copy_class_ids.push_back((long) id);
        if (class_ids.size() != tensor.getZ())
          throw std::runtime_error("Received class ids and tensor size do not match: " +
                                   std::to_string(class_ids.size()) + " vs " +
                                   std::to_string(tensor.getZ()));
        else
          mpi_put(std::to_string(i) + " >> Appending tensor");
        training_set.append(tensor.shallowCopy(), class_ids, copy_class_ids);
      }
      queue.finish();
    }

  }   // namespace

  MPITrainingController::MPITrainingController(std::filesystem::path const &output_path,
                                               nnet::Model &model, nnet::Optimizer &optimizer,
                                               TrainingCollection &training_collection,
                                               size_t max_epoch, bool output_stats)
      : Controller(output_path), model(&model), optimizer(&optimizer),
        training_collection(&training_collection), max_epoch(max_epoch),
        is_outputting_stats(output_stats) {}

  bool areTensorEqual(const math::clFTensor &tensor1, const math::clFTensor &tensor2) {
    if (tensor1.getZ() != tensor2.getZ() || tensor1.getY() != tensor2.getY() ||
        tensor1.getX() != tensor2.getX()) {
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


  ControllerResult MPITrainingController::run() {
    int rank;
    int world_size;
    std::vector<math::clFTensor> targets;
    auto &training_set = training_collection->getTrainingSet();
    InputSet input_set(training_set.getInputWidth(), training_set.getInputHeight());
    initializeMPI(rank, world_size);

    // Todo: Remove after testing
    if (rank != 0) {
      assert(training_collection != nullptr);
      assert(training_collection->getClassCount() == 0);
      assert(model != nullptr);
      assert(optimizer != nullptr);
    }
    // Todo: --------------------

    if (rank == 0) {
      scatterTrainingSet(training_collection->getTrainingSet(), input_set);
    } else {
      receiveTrainingSet(input_set);
    }
    mpi_put("Training set received, input_set size: " + std::to_string(input_set.getSize()) +
                    ", classCount: " + std::to_string(input_set.getClassIds().size()) +
                    ", tensorCount: " + std::to_string(input_set.getTensorCount()),
            tscl::Log::Trace);

    try {
      targets = setupTargets(input_set);
      mpi_put("Setup targets, size: " + std::to_string(targets.size()), tscl::Log::Trace);

      mpi_put("Starting training", tscl::Log::Trace);

      std::cout << std::endl;
    } catch (std::exception &e) {
      mpi_put("Error setting up targets: " + std::string(e.what()), tscl::Log::Error);
      return {1, "Error setting up targets"};
    }


    /*
    ModelEvaluator stats_evaluator(*model, eval_set, is_outputting_stats);

    std::future<ModelEvaluation> evaluation_future =
            std::async(std::launch::async, []() { return stats_evaluator.evaluate(); });*/


    for (size_t curr_epoch = 0; curr_epoch < max_epoch; curr_epoch++) {
      auto epoch_duration = runEpoch(*optimizer, input_set, targets);

      // printEvaluation(evaluation_future.get());
      //  Async evaluation to avoid downtime
      mpi_put("Epoch took " + std::to_string(epoch_duration.count()) + "ms",
              tscl::Log::Information);
      /*evaluation_future =
              std::async(std::launch::async, []() { return stats_evaluator.evaluate(); });*/
      return {0, "One iter successful"};
    }
    // printEvaluation(evaluation_future.get());

    return {0, "Training completed"};
  }


}   // namespace control