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
    Sample(size_t id, size_t class_id, math::clFMatrix &&data)
        : id(id), class_id(class_id), data(std::move(data)) {}

    size_t getId() const { return id; }

    size_t getClass() const { return class_id; }

    void setClass(size_t class_id) { this->class_id = class_id; }

    /**
     * @brief Returns the opencl buffer containing the data of this sample
     * Note that the data returned may be part of a larger tensor
     * @return
     */
    math::clFMatrix &getData() { return data; }

  private:
    size_t id;
    size_t class_id;
    math::clFMatrix data;
  };

  class InputSetIterator;

  /**
   * @brief A set of samples that can be used to feed a neural network model
   * Note that the samples are grouped in Tensors of heterogeneous size, which can be used for
   * batched operations.
   */
  class InputSet {
  public:
    friend InputSetIterator;

    /**
     * @brief Append the given samples and their associated tensor to the input set
     * No check is done to ensure the samples are contained in the tensor, beware that this may
     * leads to undefined behavior.
     *
     * @param tensor The tensor containing the samples
     * @param ids A vector containing the ids of the samples to append
     * @param class_id A vector containing the class ids of the samples to append. If the size of
     * this vector is smaller than the size of the ids vector, the class id is ignored. This allow
     * one to create an InputSet where the classes are not known
     */
    void append(math::clFTensor &&tensor, std::vector<size_t> ids,
                std::vector<size_t> class_id = {});

    /**
     * @brief Alter the tensors size and reorder the samples (Maintaining the same ordering) to
     * match the new tensor size.
     * This methods is not thread-safe, and invalidates all iterators.
     * Note that this methods is costly, and should be used with care.
     *
     * @param new_tensor_size The new size  of the tensors. If the tensor size is not a multiple of
     * the number of samples, the last tensor will be truncated. Tensor that already are at the
     * right size are left untouched
     */
    void alterTensors(size_t new_tensor_size);

    size_t getSize() const { return size; }

    Sample operator[](size_t index) {
      if (index > size) throw std::out_of_range("Sample index out of range");

      size_t tensor_index = index;
      size_t local_index = index;

      // Since the tensors may not be of the same size, we need to iterate through them to find the
      // right one
      for (size_t i = 0; i < tensors.size(); i++) {
        // If the tensors contains the sample, break
        if (local_index < tensors[i].getZ()) {
          tensor_index = i;
          break;
        }
        // Otherwise, update the local index
        // And move to the next tensor
        local_index -= tensors[i].getZ();
      }
      // Fetch the matrix from the tensor
      auto matrix = tensors[tensor_index].getMatrix(local_index);
      // Fetch the sample id/class_id using the index, and return the matrix
      return {ids[index], class_ids[index], std::move(matrix)};
    }

    using SampleIterator = InputSetIterator;

    SampleIterator begin();

    SampleIterator end();

    /**
     * @brief Return the number of tensors contained in this set
     * Use getSize() to get the number of samples
     * @return The number of tensors
     */
    size_t getTensorCount() const { return tensors.size(); }

    /**
     * @brief Return the tensor at the given index
     * @param index The index of the tensor to return
     * @return The tensor at the given index
     */
    math::clFTensor &getTensor(size_t index) {
      if (index > tensors.size()) throw std::out_of_range("Tensor index out of range");
      return tensors[index];
    }

    /**
     * @brief Return the tensor at the given index
     * @param index The index of the tensor to return
     * @return The tensor at the given index
     */
    const math::clFTensor &getTensor(size_t index) const {
      if (index > tensors.size()) throw std::out_of_range("Tensor index out of range");
      return tensors[index];
    }

    using TensorIterator = std::vector<math::clFTensor>::iterator;
    using TensorConstIterator = std::vector<math::clFTensor>::const_iterator;

    /**
     * @brief Return an iterator to the tensors contained in this set
     * @return
     */
    TensorIterator beginTensor() { return tensors.begin(); }
    TensorConstIterator beginTensor() const { return tensors.begin(); }

    /**
     * @brief Return an iterator to the end of the tensors contained in this set
     * @return
     */
    TensorIterator endTensor() { return tensors.end(); }
    TensorConstIterator endTensor() const { return tensors.end(); }

    /**
     * @brief Return the vector containing all the tensors
     * @return
     */
    std::vector<math::clFTensor> &getTensors() { return tensors; }

    /**
     * @brief Return the vector containing all the tensors
     * @return
     */
    const std::vector<math::clFTensor> &getTensors() const { return tensors; }

  private:
    std::vector<math::clFTensor> tensors;
    size_t size = 0;
    std::vector<size_t> ids;
    std::vector<size_t> class_ids;
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
