#include "TrainingCollectionScatterer.hpp"

using namespace control;

namespace mpiw {


  void receiveTensor(math::clFTensor &tensor, std::string &class_name, std::vector<size_t> &ids,
                     std::vector<long> &class_ids, size_t tensor_index) {
    mpi_put("> receiveTensor(...)");
    MPI_Status status;

    // Receive the tensors dimensions [OFFSET, ROWS, COLS, DEPTH]
    std::array<unsigned long, 4> tensor_dimensions{};
    MPI_Recv(tensor_dimensions.data(), 4, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, &status);
    mpi_put("Received tensor " + std::to_string(tensor_index) + " dimensions: (" +
            std::to_string(tensor_dimensions[0]) + ", " + std::to_string(tensor_dimensions[1]) +
            ", " + std::to_string(tensor_dimensions[2]) + ", " +
            std::to_string(tensor_dimensions[3]) + ")");
    size_t tensor_depth = tensor_dimensions[3];

    // Receive the tensor
    math::clFTensor tmp_tensor(tensor_dimensions.at(1), tensor_dimensions.at(2),
                               tensor_dimensions.at(3));

    math::FloatMatrix tmp_matrix(tensor_dimensions.at(1), tensor_dimensions.at(2));
    for (int z = 0; z < tensor_depth; z++) {
      MPI_Recv(tmp_matrix.getData(), (int) (tensor_dimensions.at(1) * tensor_dimensions.at(2)),
               MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      tmp_tensor[z] = math::clFMatrix(tmp_matrix, true);
    }
    mpi_put("Received entire tensor " + std::to_string(tensor_index));

    // Copy the temporary tensor to the final one
    /*cl::CommandQueue queue(cl::Context::getDefault(), cl::Device::getDefault());
    tmp_tensor.copy(tensor, queue, true);*/

    tensor = std::move(tmp_tensor);

    // Receive the samples ids
    ids.resize(tensor_depth);
    MPI_Recv(ids.data(), (int) ids.size(), MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    mpi_put("Received " + std::to_string(tensor.getDepth()) + " ids");

    // Receive the samples class ids
    class_ids.resize(tensor_depth);
    MPI_Recv(class_ids.data(), (int) class_ids.size(), MPI_LONG, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    mpi_put("Received " + std::to_string(tensor.getDepth()) + " class ids");

    // Receive class_name
    MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
    int class_name_length;
    MPI_Get_count(&status, MPI_CHAR, &class_name_length);
    class_name.resize(class_name_length);
    MPI_Recv(class_name.data(), (int) class_name.size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    mpi_put("Received class name: " + class_name);

    mpi_put("< receiveTensor(...)");
  }

  void sendCollection(size_t rank, const TrainingCollection &collection) {
    std::array<unsigned long, 2> collection_dim{};
    auto &set = send_collections.at(0).getTrainingSet();
    collection_dim = {set.getInputWidth(), set.getInputHeight()};
    assert(std::all_of(collection_dim.cbegin(), collection_dim.cend(),
                       [](const auto &e) { return e > 0; }));
    mpi_put("Sending collection dimensions: " + std::to_string(collection_dim.at(0)) + "x" +
            std::to_string(collection_dim.at(1)));
    for (int p = 1; p < nprocess; p++)
      MPI_Send(collection_dim.data(), 2, MPI_UNSIGNED_LONG, p, 0, MPI_COMM_WORLD);

    for (int p = 1; p < nprocess; p++) {
      const TrainingCollection &current_collection = send_collections.at(p);
      assert(!current_collection.getTargets().empty());

      // Send the collection size
      mpi_put("Sending collection size (" + std::to_string(current_collection.getTargets().size()) +
              ") to " + std::to_string(p));
      const unsigned long current_targets_count = current_collection.getTargets().size();
      MPI_Send(&current_targets_count, 1, MPI_UNSIGNED_LONG, p, 0, MPI_COMM_WORLD);

      for (size_t t = 0; t < current_collection.getTrainingSet().getTensorCount(); t++)
        sendTensor(current_collection.getTrainingSet(),
                   current_collection.getTrainingSet().getClasses().at(t),
                   current_collection.getTrainingSet().getTensor(t), p, t);
    }
  }

  TrainingCollection receiveCollection(size_t rank) {
    std::array<unsigned long, 2> collection_dim{};
    MPI_Recv(collection_dim.data(), 2, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    TrainingCollection recv_collection(collection_dim.at(0), collection_dim.at(1));

    unsigned long nb_tensor = 0;
    MPI_Recv(&nb_tensor, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<size_t> ids;
    std::vector<long> class_ids;
    std::vector<std::string> class_names(nb_targets);

    for (size_t t = 0; t < nb_tensor; t++) {
      math::clFTensor tensor;
      receiveTensor(tensor, class_names.at(t), ids, class_ids, t);
      recv_collection.getTrainingSet().append(std::move(tensor), ids, class_ids);
    }

    recv_collection.getTrainingSet().updateClasses(class_names);
    recv_collection.makeTrainingTargets();
    return recv_collection;
  }

  TrainingCollection TrainingCollectionScatterer::scatter(TrainingCollection global_collection) {
    std::vector<TrainingCollection> collections = global_collection.split(size);

    if (collections.size() != size) {
      throw std::runtime_error("TrainingCollectionScatterer::scatter: number of collections is "
                               "not equal to number of processes");
    }

    for (size_t prank = 1; prank < size; ++prank) { sendCollection(prank, collections[prank]); }

    return std::move(collections[0]);
  }

  control::TrainingCollection TrainingCollectionScatterer::receive() {
    return receiveCollection(0);
  }
}   // namespace mpiw