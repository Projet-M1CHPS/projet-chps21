#include "InputSet.hpp"
#include "image/Image.hpp"

namespace control {
  namespace {
    void fillTensor(math::clFTensor &tensor, std::vector<math::clFTensor> &old_tensors,
                    size_t &local_index, cl::CommandQueue &queue) {
      // Fill the new tensor up
      for (size_t i = 0; i < tensor.getDepth(); i++) {
        // Copy the old matrix into the new one
        // We cannot copy the whole tensor because the size of the new tensor might be different
        auto old_matrix = old_tensors.front()[local_index];
        auto new_matrix = tensor[i];

        // Perform a non-blocking copy
        queue.enqueueCopyBuffer(old_matrix.getBuffer(), new_matrix.getBuffer(), 0, 0,
                                sizeof(float) * tensor.getRows() * tensor.getCols());

        // Increment the local index and check if we need to jump to the next tensor
        if (local_index++ >= old_tensors[0].getDepth()) {
          // We don't need the old tensors anymore
          // Removing it allows us to reuse the memory
          // As soon as the copy is finished
          old_tensors.erase(old_tensors.begin());
          local_index = 0;
          // Since we don't want to overfill the memory, we wait until all copies are finished
          // So that the old tensors are removed before moving on
          queue.finish();
        }
      }
    }
  }   // namespace

  InputSet &InputSet::operator=(InputSet &&other) noexcept {
    std::shared_lock<std::shared_mutex> lock(other.mutex);
    std::unique_lock<std::shared_mutex> lock2(mutex);

    input_width = other.input_width;
    input_height = other.input_height;

    tensors = std::move(other.tensors);
    samples = std::move(other.samples);
    class_names = std::move(other.class_names);
    return *this;
  }

  void InputSet::append(math::clFTensor &&tensor, const std::vector<size_t> &new_ids,
                        const std::vector<long> &class_ids) {
    std::scoped_lock lock(mutex);

    // Each sample must have its own id
    if (tensor.getDepth() != new_ids.size()) {
      std::cerr << "Error: tensor.getDepth(): " << tensor.getDepth()
                << " != new_ids.size(): " << new_ids.size() << std::endl;
      throw std::runtime_error("InputSet::append: tensor and new_ids must have same size");
    }

    // The input tensor should be correctly sized
    if (tensor.getRows() != input_width or tensor.getCols() != input_height) {
      std::cerr << "Error: tensor.getRows(): " << tensor.getRows()
                << " != input_width: " << input_width << std::endl;
      std::cerr << "Error: tensor.getCols(): " << tensor.getCols()
                << " != input_height: " << input_height << std::endl;
      throw std::runtime_error(
              "InputSet::append: tensor must have same size as input_width and input_height");
    }

    for (size_t i = 0; i < new_ids.size(); i++) {
      long class_id = -1;
      if (class_ids.size() == new_ids.size()) class_id = class_ids[i];
      samples.emplace_back(new_ids[i], class_id, tensor[i]);
    }
    tensors.push_back(std::move(tensor));
  }

  void InputSet::alterTensors(size_t new_tensor_size) {
    std::scoped_lock lock(mutex);

    std::vector<math::clFTensor> new_tensors;
    size_t size = samples.size();
    size_t new_tensor_count = size / new_tensor_size;
    size_t new_tensor_remainder = size % new_tensor_size;

    size_t local_index = 0;

    cl::CommandQueue queue(utils::cl_wrapper.getContext(), utils::cl_wrapper.getDefaultDevice());

    for (size_t i = 0; i < new_tensor_count; ++i) {
      // Create a new tensor
      new_tensors.emplace_back(input_width, input_height, new_tensor_size);
      // Fill the new tensor up, and erase old tensors that have been depleted to make room
      fillTensor(new_tensors.back(), tensors, local_index, queue);
    }

    // Handle the remainder
    new_tensors.emplace_back(input_width, input_height, new_tensor_remainder);
    fillTensor(new_tensors.back(), tensors, local_index, queue);
    queue.finish();
  }

  void InputSet::removeTensors(size_t start, size_t end) {
    std::scoped_lock lock(mutex);

    if (start >= end) {
      throw std::runtime_error("InputSet::removeTensors: start must be smaller than end");
    }

    if (end > tensors.size()) {
      throw std::runtime_error("InputSet::removeTensors: end must be smaller than tensors.size()");
    }

    size_t first_global_index = 0;
    size_t last_global_index = 0;

    for (size_t i = 0; i < end; i++) {
      if (i < start) { first_global_index += tensors[i].getDepth(); }
      last_global_index += tensors[i].getDepth();
    }
    samples.erase(samples.begin() + first_global_index, samples.begin() + last_global_index);
    // Remove the tensors from the vector
    tensors.erase(tensors.begin() + start, tensors.begin() + end);
  }

  void InputSet::shuffle(size_t random_seed) {
    std::scoped_lock lock(mutex);

    // Since we need to shuffle not only the tensors, but their content
    // The easiest way is to recreate a new input set and move copy the samples
    std::mt19937 generator(random_seed);
    // We shuffle the samples
    std::shuffle(samples.begin(), samples.end(), generator);

    InputSet buffer(input_width, input_height);

    // We re-create the input set, using the exact same tensors size, but filling them with the
    // shuffled samples
    size_t sample_index = 0;
    cl::CommandQueue queue(utils::cl_wrapper.getContext(), utils::cl_wrapper.getDefaultDevice());
    for (const auto &tensor : tensors) {
      size_t tensor_size = tensor.getDepth();
      math::clFTensor buffer_tensor(input_width, input_height, tensor_size);

      std::vector<size_t> new_ids(tensor_size);
      std::vector<long> class_ids(tensor_size);

      for (auto &matrix : buffer_tensor.getMatrices()) {
        matrix.copy(samples[sample_index].getData(), queue, false);

        new_ids[sample_index] = samples[sample_index].getId();
        class_ids[sample_index] = samples[sample_index].getClass();
        sample_index++;
      }
      buffer.append(std::move(buffer_tensor), new_ids, class_ids);
    }
    queue.finish();
    // We now delete the old set and move copy the new one
    *this = std::move(buffer);
  }


  void InputSet::shuffle() { shuffle(std::random_device{}()); }

  void InputSet::split(size_t nb_sets, std::vector<InputSet> &sub_sets) const {
    assert(sub_sets.empty());
    std::cout << "InputSet::split() Splitting InputSet intro " + std::to_string(nb_sets) +
                         " sub-sets."
              << std::endl;

    auto sub_classes = splitClassNames(nb_sets);

    std::vector<size_t> tensor_counts(nb_sets);
    size_t part = getTensorCount() / nb_sets;
    int rest = ((int) getTensorCount()) % nb_sets;
    for (size_t i = 0; i < nb_sets; i++) tensor_counts[i] = part + ((rest-- > 0) ? 1 : 0);

    std::cout << "Number of classes: " << getSamplesClassIds().size() << std::endl;
    size_t total_matrix_count = 0;
    for (auto &item : tensors) total_matrix_count += item.getDepth();
    std::cout << "Number of matrices: " << total_matrix_count << std::endl;

    // Create the new sets
    size_t current_tensor_index = 0;
    size_t current_sample_index = 0;
    for (size_t i = 0; i < nb_sets; i++) {
      InputSet current_set(input_width, input_height);
      std::cout << "InputSet::split() Creating sub-set " + std::to_string(i) + "." << std::endl;
      std::cout << "InputSet::split() Sub-set " + std::to_string(i) +
                           " dimensions: " + std::to_string(current_set.getInputWidth()) + "x" +
                           std::to_string(current_set.getInputHeight()) + "x" +
                           std::to_string(current_set.getTensorCount())
                << std::endl;

      std::cout << "InputSet::split() Dimensions of sub-sets: " << std::endl;
      for (auto &item : sub_sets)
        std::cout << "InputSet::split() " << item.getInputWidth() << "x" << item.getInputHeight()
                  << std::endl;
      for (size_t j = 0; j < tensor_counts.at(i); j++) {
        auto &item = getTensor(current_tensor_index++);
        std::vector<size_t> new_ids(item.getDepth());
        std::vector<long> new_class_ids(item.getDepth());
        for (size_t k = 0; k < item.getDepth(); k++) {
          new_ids[k] = samples.at(current_sample_index).getId();
          new_class_ids[k] = samples.at(current_sample_index).getClass();
          current_sample_index++;
        }
        current_set.append(item.shallowCopy(), new_ids, new_class_ids);
      }
      current_set.updateClasses(sub_classes.at(i));
      assert(!current_set.getSamples().empty());
      sub_sets.emplace_back(std::move(current_set));
      std::cout << "InputSet::split() Dimensions of sub-sets: " << std::endl;
      for (auto &item : sub_sets)
        std::cout << "InputSet::split() " << item.getInputWidth() << "x" << item.getInputHeight()
                  << std::endl;
    }

    std::cout << "InputSet::split() Done." << std::endl;
    std::cout << "InputSet::split() Number of sub-sets: " << sub_sets.size() << std::endl;
    std::cout << "InputSet::split() Dimensions of sub-sets: " << std::endl;
    for (auto &item : sub_sets)
      std::cout << "InputSet::split() " << item.getInputWidth() << "x" << item.getInputHeight()
                << std::endl;
  }

  std::vector<std::vector<std::string>> InputSet::splitClassNames(size_t nb_sets) const {
    std::vector<std::vector<std::string>> new_class_names(nb_sets);
    size_t classes_per_set = class_names.size() / nb_sets;
    int classes_left = ((int) class_names.size()) % nb_sets;

    size_t acc_class_index = 0;
    for (size_t i = 0; i < nb_sets; i++) {
      size_t count = classes_per_set + (classes_left-- > 0 ? 1 : 0);
      for (size_t j = 0; j < count; j++)
        new_class_names.at(i).push_back(class_names[acc_class_index++]);
    }

    return new_class_names;
  }

  std::vector<std::vector<math::clFTensor>> InputSet::splitTensors(size_t nb_sets) const {
    assert(nb_sets <= getTensorCount());
    assert(nb_sets > 0);
    std::vector<std::vector<math::clFTensor>> new_tensors(nb_sets);

    // Amount of tensors per set
    size_t part = getTensorCount() / nb_sets;
    int remainder = ((int) getTensorCount()) % nb_sets;
    assert(part > 0);
    for (size_t i = 0; i < nb_sets; i++) new_tensors[i].resize(part + (remainder-- > 0 ? 1 : 0));

    // OpenCl queues (todo: split in threads, each with their own queue)
    std::vector<cl::CommandQueue> queues(nb_sets);
    for (size_t i = 0; i < nb_sets; i++)
      queues[i] = cl::CommandQueue(utils::cl_wrapper.getContext(),
                                   utils::cl_wrapper.getDefaultDevice());

    // Split the tensors
    size_t tensor_index = 0;
    for (size_t i = 0; i < nb_sets; i++) {
      for (size_t j = 0; j < new_tensors.at(i).size(); j++)
        new_tensors[i][j].copy(tensors[tensor_index++], queues.at(i), true);
      tensor_index++;
    }

    // Wait for all queues to finish
    for (auto &queue : queues) queue.finish();

    return new_tensors;
  }

  const std::vector<Sample> &InputSet::getSamples() const { return samples; }

}   // namespace control