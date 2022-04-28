#include "TrainingCollectionScatterer.hpp"

#include <utility>

using namespace control;

namespace mpiw {
  namespace {

    class ImagePack {
    public:
      ImagePack() = default;

      ImagePack(const math::clFTensor &tensor, std::vector<size_t> ids, std::vector<long> class_ids)
          : tensor(tensor.shallowCopy()), ids(std::move(ids)), class_ids(std::move(class_ids)) {}

      void send(size_t destination, MPI_Comm comm) const {
        if (tensor.getDepth() != ids.size() or ids.size() != class_ids.size() or
            tensor.getDepth() == 0)
          throw std::runtime_error("ImagePack::send(): Invalid image pack");

        cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();

        // Map the tensor to the memory so we can send the tensor using DMA
        float *mapped_data =
                (float *) queue.enqueueMapBuffer(tensor.getBuffer(), CL_TRUE, CL_MAP_READ,
                                                 tensor.getOffsetInBytes(), tensor.sizeInBytes());

        MPI_Send(mapped_data, tensor.size(), MPI_FLOAT, destination, 0, comm);
        // Don't forget to unmap the tensor
        queue.enqueueUnmapMemObject(tensor.getBuffer(), mapped_data);

        // Send the samples ids and classes
        MPI_Send(ids.data(), ids.size(), MPI_UNSIGNED_LONG, destination, 0, comm);
        MPI_Send(class_ids.data(), class_ids.size(), MPI_LONG, destination, 0, comm);

        // Ensure the unmap is finished before returning
        queue.finish();
      }

      static ImagePack receive(size_t source, MPI_Comm comm, size_t height, size_t width) {
        if (height == 0 or width == 0)
          throw std::runtime_error("ImagePack::receive(): Invalid image pack");

        int tensor_size = 0;
        MPI_Status status;
        MPI_Probe(source, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_FLOAT, &tensor_size);

        if (tensor_size == 0) throw std::runtime_error("ImagePack::receive(): Invalid image pack");

        // Deduce the depth of the tensor by dividing the number of elements by the number of
        // element in a single matrix
        size_t single_matrix_size = height * width;
        int depth = tensor_size / single_matrix_size;

        cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();
        math::clFTensor tensor(width, height, depth);

        // Map the tensor to the memory so we can receive the tensor using DMA
        float *mapped_data = (float *) utils::cl_wrapper.getDefaultQueue().enqueueMapBuffer(
                tensor.getBuffer(), CL_TRUE, CL_MAP_WRITE, tensor.getOffsetInBytes(),
                tensor.sizeInBytes());
        MPI_Recv(mapped_data, tensor.size(), MPI_FLOAT, source, 0, comm, MPI_STATUS_IGNORE);

        // Don't forget to unmap the tensor
        queue.enqueueUnmapMemObject(tensor.getBuffer(), mapped_data);

        // Receive the samples ids and classes
        std::vector<size_t> ids(depth);
        MPI_Recv(ids.data(), ids.size(), MPI_UNSIGNED_LONG, source, 0, comm, MPI_STATUS_IGNORE);

        std::vector<long> class_ids(depth);
        MPI_Recv(class_ids.data(), class_ids.size(), MPI_LONG, source, 0, comm, MPI_STATUS_IGNORE);

        // Build the image pack
        ImagePack res;
        res.tensor = std::move(tensor);
        res.ids = std::move(ids);
        res.class_ids = std::move(class_ids);

        return res;
      }

      static void sendPackCount(size_t destination, MPI_Comm comm, size_t pack_count) {
        MPI_Send(&pack_count, 1, MPI_UNSIGNED_LONG, destination, 0, comm);
      }

      static size_t receivePackCount(size_t source, MPI_Comm comm) {
        size_t res = 0;
        MPI_Recv(&res, 1, MPI_UNSIGNED_LONG, source, 0, comm, MPI_STATUS_IGNORE);
        return res;
      }

      math::clFTensor &getTensor() { return tensor; }
      std::vector<size_t> &getIds() { return ids; }
      std::vector<long> &getClassIds() { return class_ids; }

    private:
      math::clFTensor tensor;
      std::vector<size_t> ids;
      std::vector<long> class_ids;
    };

    void sendCollection(size_t target_rank, MPI_Comm comm, const TrainingCollection &collection) {
      const auto &dataset = collection.getTrainingSet();

      size_t tensor_count = dataset.getTensorCount();
      // Let the receiver know about the number of tensors that it will receive
      ImagePack::sendPackCount(target_rank, comm, dataset.getTensorCount());
      std::cout << "Sending " << tensor_count << " tensors to rank " << target_rank << std::endl;

      size_t sample_index = 0;
      for (size_t i = 0; i < tensor_count; i++) {
        const auto &tensor = dataset.getTensor(i);

        // Build vectors containing the ids and classes of the samples
        std::vector<size_t> ids(tensor.getDepth(), 0);
        std::vector<long> class_ids(tensor.getDepth(), -1);

        for (size_t j = 0; j < tensor.getDepth(); j++) {
          ids[j] = dataset.getSampleId(sample_index);
          class_ids[j] = dataset.getClassOf(sample_index);
          sample_index++;
        }

        ImagePack pack(tensor, ids, class_ids);
        pack.send(target_rank, comm);
      }
    }

    TrainingCollection receiveCollection(int source, MPI_Comm comm, size_t height, size_t width,
                                         const std::vector<std::string> &classes) {
      TrainingCollection recv_collection(height, width);
      recv_collection.updateClasses(classes);

      // Receive the number of tensors in our collection
      const size_t nb_tensor = ImagePack::receivePackCount(source, comm);
      std::cout << "Receiving " << nb_tensor << " tensors from rank " << source << std::endl;

      // Receive each tensor individually
      // We only need to receive the training set
      for (size_t t = 0; t < nb_tensor; t++) {
        auto image_pack = ImagePack::receive(source, comm, height, width);
        // Append the tensor at the end of the training set
        recv_collection.getTrainingSet().append(std::move(image_pack.getTensor()),
                                                image_pack.getIds(), image_pack.getClassIds());
      }
      return recv_collection;
    }

    void broadcastMetadata(MPI_Comm comm, size_t width, size_t height,
                           const std::vector<std::string> &classes) {
      // Send the dimensions
      std::array<size_t, 3> dims = {height, width, classes.size()};
      MPI_Bcast(dims.data(), 3, MPI_UNSIGNED_LONG, 0, comm);

      // Send every class names separately
      for (const auto &c : classes) {
        size_t len = c.size() + 1;
        MPI_Bcast(&len, 1, MPI_UNSIGNED_LONG, 0, comm);
        MPI_Bcast((void *) c.data(), c.size() + 1, MPI_CHAR, 0, comm);
      }
    }

    std::tuple<size_t, size_t, std::vector<std::string>> broadcastMetadata(MPI_Comm comm) {
      // Receive the images dimensions, as well as the number of classes
      std::array<size_t, 3> dims = {0, 0, 0};
      MPI_Bcast(dims.data(), 3, MPI_UNSIGNED_LONG, 0, comm);

      std::vector<std::string> classes;
      classes.reserve(dims[2]);

      // We cannot receive a vector of string, so we have to treat them separately
      for (size_t i = 0; i < dims[2]; i++) {
        // Receive the length of the string
        MPI_Status status;

        size_t string_len = 0;
        MPI_Bcast(&string_len, 1, MPI_UNSIGNED_LONG, 0, comm);

        std::string class_name(string_len, '\0');
        MPI_Bcast((void *) class_name.data(), class_name.size(), MPI_CHAR, 0, comm);
        classes.push_back(class_name);
      }
      return {dims[0], dims[1], classes};
    }
  }   // namespace

  TrainingCollectionScatterer::TrainingCollectionScatterer(MPI_Comm comm) : comm(comm) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
  }

  TrainingCollection
  TrainingCollectionScatterer::scatter(const TrainingCollection &global_collection) {
    // Split the global collection into smaller parts
    std::vector<TrainingCollection> collections = global_collection.splitTrainingSet(size);

    // We can't have a process with no data, or can we ?
    if (collections.size() != size) {
      throw std::runtime_error("TrainingCollectionScatterer::scatter: number of collections is "
                               "not equal to number of processes");
    }

    // Send the images width and height, as well as the classes names
    const auto &dataset = global_collection.getTrainingSet();
    broadcastMetadata(comm, dataset.getInputHeight(), dataset.getInputHeight(),
                      global_collection.getClassNames());


    for (size_t process_rank = 1; process_rank < size; ++process_rank) {
      sendCollection(process_rank, comm, collections[process_rank]);
    }

    return std::move(collections[0]);
  }

  control::TrainingCollection TrainingCollectionScatterer::receive(int source) {
    if (rank == source) {
      throw std::runtime_error("TrainingCollectionScatterer::receive: cannot receive from self");
    }

    if (source > size) {
      throw std::runtime_error("TrainingCollectionScatterer::receive: invalid id for source");
    }

    // Receive the metadata from the master
    auto [height, width, classes] = broadcastMetadata(comm);

    return receiveCollection(source, comm, height, width, classes);
  }
}   // namespace mpiw