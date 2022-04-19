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
      throw std::runtime_error("InputSet::append: tensor and new_ids must have same size");
    }

    // The input tensor should be correctly sized
    if (tensor.getRows() != input_width or tensor.getCols() != input_height) {
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
}   // namespace control