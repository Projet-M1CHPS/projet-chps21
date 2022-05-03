
#include "MPIMLPModel.hpp"

namespace nnet {
  namespace {
    void synchronizeWeightsAndBiases(std::unique_ptr<MPIMLPModel> &model) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      std::vector<void *> biases_ptr, weights_ptr;
      // Map the biases asynchronously
      for (auto &bias : model->getPerceptron().getBiases())
        biases_ptr.push_back(enqueueMapBuffer(bias.getBuffer(), CL_FALSE,
                                              rank == 0 ? CL_MAP_READ : CL_MAP_WRITE,
                                              bias.getOffsetInBytes(), bias.sizeInBytes()));
      // Map the weights asynchronously
      for (auto &weight : model->getPerceptron().getWeights())
        weights_ptr.push_back(enqueueMapBuffer(weight.getBuffer(), CL_FALSE,
                                               rank == 0 ? CL_MAP_READ : CL_MAP_WRITE,
                                               weight.getOffsetInBytes(), weight.sizeInBytes()));

      // Wait for all the mappings to complete
      cl::CommandQueue::getDefault().finish();

      std::vector<MPI_Request> requests;

      // Synchronize the biases with the other processes
      for (size_t i = 0; i < biases_ptr.size(); i++) {
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Ibcast(biases_ptr[i], (int) model->getPerceptron().getBiases()[i].size(), MPI_FLOAT, 0,
                   MPI_COMM_WORLD, &requests.back());
      }

      // Synchronize the weights with the other processes
      for (size_t i = 0; i < weights_ptr.size(); i++) {
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Ibcast(weights_ptr[i], (int) model->getPerceptron().getWeights()[i].size(), MPI_FLOAT,
                   0, MPI_COMM_WORLD, &requests.back());
      }

      // Wait for all the synchronizations to complete
      MPI_Waitall((int) requests.size(), requests.data(), MPI_STATUSES_IGNORE);

      // Unmap the biases
      for (size_t i = 0; i < biases_ptr.size(); i++)
        enqueueUnmapMemObject(model->getPerceptron().getBiases()[i].getBuffer(), biases_ptr[i]);
      // Unmap the weights
      for (size_t i = 0; i < weights_ptr.size(); i++)
        enqueueUnmapMemObject(model->getPerceptron().getWeights()[i].getBuffer(), weights_ptr[i]);

      cl::CommandQueue::getDefault().finish();
    }
  }   // namespace

  std::unique_ptr<MPIMLPModel> MPIMLPModel::random(MLPTopology const &topology,
                                                   af::ActivationFunctionType af) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto res = std::make_unique<MPIMLPModel>();
    auto &mlp = res->getPerceptron();
    mlp.setTopology(topology);
    mlp.setActivationFunction(af);

    // Only the root process will initialize the weights & biases
    if (rank == 0) mlp.randomizeWeight();

    tscl::logger("[P" + std::to_string(rank) + "]: Synchronizing weights and biases...");
    synchronizeWeightsAndBiases(res);
    tscl::logger("[P" + std::to_string(rank) + "]: Synchronization complete.");

    return res;
  }
}   // namespace nnet