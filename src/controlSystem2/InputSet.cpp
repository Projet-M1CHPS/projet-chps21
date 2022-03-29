#include "InputSet.hpp"

namespace control {
  namespace {
    void fillTensor(math::clFTensor &tensor, std::vector<math::clFTensor> &old_tensors,
                    size_t &local_index, cl::CommandQueue &queue) {
      // Fill the new tensor up
      for (size_t i = 0; i < tensor.getZ(); i++) {
        // Copy the old matrix into the new one
        // We cannot copy the whole tensor because the size of the new tensor might be different
        auto old_matrix = old_tensors.front().getMatrix(local_index);
        auto new_matrix = tensor.getMatrix(i);

        // Perform a non-blocking copy
        queue.enqueueCopyBuffer(old_matrix.getBuffer(), new_matrix.getBuffer(), 0, 0,
                                sizeof(float) * tensor.getX() * tensor.getY());

        // Increment the local index and check if we need to jump to the next tensor
        if (local_index++ >= old_tensors[0].getZ()) {
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

  void InputSet::append(math::clFTensor &&tensor, const std::vector<size_t> &new_ids,
                        const std::vector<long> &class_id) {
    // Each sample must have its own id
    if (tensor.getZ() != new_ids.size()) {
      throw std::runtime_error("InputSet::append: tensor and new_ids must have same size");
    }

    // The input tensor should be correctly sized
    if (tensor.getX() != input_width or tensor.getY() != input_height) {
      throw std::runtime_error(
              "InputSet::append: tensor must have same size as input_width and input_height");
    }


    ids.insert(ids.end(), new_ids.begin(), new_ids.end());

    // Class ids are optional, and are set to -1 if not provided
    if (class_id.size() == new_ids.size()) {
      class_ids.insert(class_ids.end(), class_id.begin(), class_id.end());
    } else if (class_id.empty()) {
      class_ids.insert(class_ids.end(), new_ids.size(), -1);
    } else {
      throw std::runtime_error("InputSet::append: class_id.size() != new_ids.size()");
    }

    tensors.push_back(std::move(tensor));
  }

  void InputSet::alterTensors(size_t new_tensor_size) {
    std::vector<math::clFTensor> new_tensors;
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

  void InputSet::shuffle(size_t random_seed) {
    // Shuffle each vectors with the same seed to maintain the same order
    std::mt19937 generator(random_seed);
    auto gen2 = generator;
    auto gen3 = generator;

    std::shuffle(ids.begin(), ids.end(), generator);
    std::shuffle(class_ids.begin(), class_ids.end(), gen2);
    std::shuffle(tensors.begin(), tensors.end(), gen3);
  }

}   // namespace control