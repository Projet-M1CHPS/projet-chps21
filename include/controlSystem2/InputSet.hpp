#pragma once
#include "clUtils/clFMatrix.hpp"
#include "clUtils/clFTensor.hpp"
#include <filesystem>
#include <iostream>
#include <utility>

namespace control {

  /**
   * @brief A sample of input data.
   */
  class Sample {
  public:
    /**
     * @brief Builds a new sample
     * @param id The unique id of this sample
     * @param class_id The id of the class of this sample
     * @param data The raw data of the sample
     */
    Sample(size_t id, long class_id, math::clFMatrix &&data)
        : id(id), class_id(class_id), data(std::move(data)) {}

    size_t getId() const { return id; }

    size_t getClass() const { return class_id; }

    void setClass(long new_id) { class_id = new_id; }

    /**
     * @brief Returns the opencl buffer containing the data of this sample
     * Note that the data returned may be part of a larger tensor
     * @return
     */
    math::clFMatrix &getData() { return data; }

  private:
    size_t id;
    long class_id;
    math::clFMatrix data;
  };

  class InputSetIterator;

  /**
   * @brief Thread-safe set of samples that can be used to feed a neural network model
   * Note that the samples are grouped in Tensors of heterogeneous size, which can be used for
   * batched operations.
   *
   * Does not support removing random samples from the set (Really complex operation since this
   * would require tensors to be rearranged, and barely needed). However, supports removal of entire
   * tensors.
   */
  class InputSet final {
  public:
    friend InputSetIterator;

    InputSet(size_t input_width, size_t input_height)
        : input_width(input_width), input_height(input_height) {}

    /**
     * @brief Append the given samples and their associated tensor to the input set. Samples are
     * added at the end of the set.
     *
     * @param tensor The tensor containing the samples
     * @param ids A vector containing the ids of the samples to append
     * @param class_id A vector containing the class ids of the samples to append. If the size of
     * this vector is smaller than the size of the ids vector, the class id is ignored. This allow
     * one to create an InputSet where the classes are not known
     */
    void append(math::clFTensor &&tensor, const std::vector<size_t> &ids,
                const std::vector<long> &class_id = {});

    /**
     * @brief Alter the tensors size and reorder the samples (Maintaining the same ordering) to
     * match the new tensor size.
     * This methods is not thread-safe, and invalidates all iterators.
     * Note that this methods is costly, and should be used with care.
     *
     * @param new_tensor_size The new size  of the tensors. If the tensor size is not a multiple of
     * the number of samples, the last tensor will be truncated.
     */
    void alterTensors(size_t new_tensor_size);

    /**
     * @brief Remove tensors in [start, end[. This operation is thread-safe, but invalidates any
     * iterators.
     * @param start The index of the first tensor to remove
     * @param end The index of the last tensor (not included) to remove
     */
    void removeTensors(size_t start, size_t end);

    /**
     * @brief Shuffle the samples in the set. This operation is thread-safe, but invalidates any
     * iterators
     * @param random_seed
     */
    void shuffle(size_t random_seed);

    /**
     * @brief Shuffle the samples in the set using a random seed. This operation is thread-safe, but
     * invalidates any iterators
     */
    void shuffle();

    size_t getSize() const {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return size;
    }

    size_t getInputWidth() const { return input_width; }
    size_t getInputHeight() const { return input_height; }

    Sample operator[](size_t index);

    using SampleIterator = InputSetIterator;

    SampleIterator begin();

    SampleIterator end();

    /**
     * @brief Return the number of tensors contained in this set
     * Use getSize() to get the number of samples
     * @return The number of tensors
     */
    size_t getTensorCount() const {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return tensors.size();
    }

    /**
     * @brief Return the tensor at the given index
     * @param index The index of the tensor to return
     * @return The tensor at the given index
     */
    math::clFTensor &getTensor(size_t index) {
      std::shared_lock<std::shared_mutex> lock(mutex);
      if (index > tensors.size()) throw std::out_of_range("Tensor index out of range");
      return tensors[index];
    }

    /**
     * @brief Return the tensor at the given index
     * @param index The index of the tensor to return
     * @return The tensor at the given index
     */
    const math::clFTensor &getTensor(size_t index) const {
      std::shared_lock<std::shared_mutex> lock(mutex);
      if (index > tensors.size()) throw std::out_of_range("Tensor index out of range");
      return tensors[index];
    }

    using TensorIterator = std::vector<math::clFTensor>::iterator;
    using TensorConstIterator = std::vector<math::clFTensor>::const_iterator;

    /**
     * @brief Return an iterator to the tensors contained in this set
     * @return
     */
    TensorIterator beginTensor() {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return tensors.begin();
    }

    TensorConstIterator beginTensor() const {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return tensors.begin();
    }

    /**
     * @brief Return an iterator to the end of the tensors contained in this set
     * @return
     */
    TensorIterator endTensor() {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return tensors.end();
    }

    TensorConstIterator endTensor() const {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return tensors.end();
    }

    /**
     * @brief Return the vector containing all the tensors
     * @return
     */
    std::vector<math::clFTensor> &getTensors() {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return tensors;
    }

    /**
     * @brief Return the vector containing all the tensors
     * @return
     */
    const std::vector<math::clFTensor> &getTensors() const {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return tensors;
    }

    /**
     * @brief Return the class id of the sample at the given index
     * @param global_index The index of the sample in the input set
     * @return The class id of the sample, -1 if it is undefined
     * Throw on out of range
     */
    long getClass(size_t global_index) const {
      std::shared_lock lock(mutex);
      if (global_index > size) throw std::out_of_range("Sample index out of range");
      return class_ids[global_index];
    }

    /**
     * TODO: Improve class / label handling
     * @brief Update the classes of the samples in the input set. Note that this method only update
     * the names of the classes, and doesn't handle the case where there are fewer classes than
     * referenced by the samples.
     * @param classes
     */
    void updateClasses(std::vector<std::string> classes) {
      std::scoped_lock lock(mutex);
      class_names = std::move(classes);
    }

  private:
    size_t size = 0;
    size_t input_width = 0, input_height = 0;

    std::vector<size_t> ids;
    std::vector<math::clFTensor> tensors;

    std::vector<long> class_ids;
    std::vector<std::string> class_names;

    mutable std::shared_mutex mutex;
  };

  /**
   * @brief Provides an iterator over the samples of an input set
   * The iterator is a bidirectional iterator, and can be used to iterate over the samples of an
   * input set with a low overhead.
   */
  class InputSetIterator {
  public:
    /**
     * @brief Creates a new iterator on the given input set, starting at the given index
     * @param parent The input set to iterate over
     * @param global_index The index of the sample to start the iteration at
     */
    InputSetIterator(InputSet &parent, size_t global_index)
        : parent(&parent), global_index(global_index) {
      // Find the tensor containing the sample
      for (size_t i = 0; i < parent.getTensorCount(); i++) {
        if (global_index < parent.getTensor(i).getZ()) {
          tensor_index = i;
          local_index = global_index;
          break;
        } else {
          global_index -= parent.getTensor(i).getZ();
        }
      }
    }

    /**
     * @brief Return the current sample
     * @return
     */
    Sample operator*() const {
      if (global_index >= parent->getSize()) throw std::out_of_range("Sample index out of range");

      auto matrix = parent->tensors[tensor_index].getMatrix(local_index);
      return {parent->ids[global_index], parent->class_ids[global_index], std::move(matrix)};
    }

    InputSetIterator operator+(long int n) const {
      InputSetIterator res(*parent, global_index, tensor_index, local_index);
      return res += n;
    }

    InputSetIterator operator-(long int n) const {
      InputSetIterator res(*parent, global_index, tensor_index, local_index);
      return res -= n;
    }

    InputSetIterator &operator+=(long int n) {
      global_index += n;
      local_index += n;
      while (local_index >= parent->tensors[tensor_index].getZ()) {
        local_index -= parent->tensors[tensor_index].getZ();
        tensor_index++;
      }
      return *this;
    }

    InputSetIterator &operator-=(long int n) {
      global_index -= n;
      local_index -= n;
      while (local_index < 0) {
        tensor_index--;
        local_index += parent->tensors[tensor_index].getZ();
      }
      return *this;
    }

    InputSetIterator &operator++() { return *this += 1; }

    InputSetIterator &operator--() { return *this -= 1; }

    bool operator==(const InputSetIterator &other) const {
      return global_index == other.global_index && tensor_index == other.tensor_index &&
             local_index == other.local_index;
    }

    bool operator!=(const InputSetIterator &other) const { return not(*this == other); }

    bool operator<(const InputSetIterator &other) const {
      return global_index < other.global_index;
    }

    bool operator>(const InputSetIterator &other) const {
      return global_index > other.global_index;
    }

    bool operator<=(const InputSetIterator &other) const {
      return global_index <= other.global_index;
    }

    bool operator>=(const InputSetIterator &other) const {
      return global_index >= other.global_index;
    }

  private:
    InputSetIterator(InputSet &parent, size_t global_index, size_t tensor_index, size_t local_index)
        : parent(&parent), global_index(global_index), tensor_index(tensor_index),
          local_index(local_index) {}

    InputSet *parent;
    size_t global_index;
    size_t local_index;
    size_t tensor_index;
  };
}   // namespace control
