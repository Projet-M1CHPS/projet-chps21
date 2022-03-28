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
     * @param path The path of the file this sample was loaded from
     * @param data The raw data of the sample
     */
    Sample(size_t id, size_t class_id, std::filesystem::path path, math::clFMatrix &&data)
        : id(id), class_id(class_id), path(std::move(path)), data(std::move(data)) {}

    size_t getId() const { return id; }

    size_t getClass() const { return class_id; }

    void setClass(size_t class_id) { this->class_id = class_id; }

    const std::filesystem::path &getPath() const { return path; }

    /**
     * @brief Returns the opencl buffer containing the data of this sample
     * Note that the data returned may be part of a larger tensor
     * @return
     */
    math::clFMatrix &getData() { return data; }

  private:
    size_t id;
    size_t class_id;
    std::filesystem::path path;
    math::clFMatrix data;
  };

  /**
   * @brief A set of samples that can be used to feed a neural network model
   * Note that the samples are grouped in Tensors of heterogeneous size, which can be used for
   * batched operations.
   */
  class InputSet {
  public:
    /**
     * @brief Append the given samples and their associated tensor to the input set
     * No check is done to ensure the samples are contained in the tensor, beware that this may
     * leads to undefined behavior.
     *
     * @param samples The samples to append to this set
     * @param tensor The tensor containing the samples
     */
    void append(std::vector<Sample> &&samples, math::clFTensor &&tensor);

    Sample &operator[](size_t index) {
      if (index > samples.size()) throw std::out_of_range("Sample index out of range");
      return samples[index];
    }

    const Sample &operator[](size_t index) const {
      if (index > samples.size()) throw std::out_of_range("Sample index out of range");
      return samples[index];
    }

    std::vector<Sample> &getSamples() { return samples; }
    const std::vector<Sample> &getSamples() const { return samples; }

    using SampleIterator = std::vector<Sample>::iterator;
    using SampleConstIterator = std::vector<Sample>::const_iterator;

    SampleIterator begin() { return samples.begin(); }

    SampleIterator end() { return samples.end(); }

    SampleConstIterator begin() const { return samples.begin(); }

    SampleConstIterator end() const { return samples.end(); }

    size_t getTensorCount() const { return tensors.size(); }

    math::clFTensor &getTensor(size_t index) {
      if (index > tensors.size()) throw std::out_of_range("Tensor index out of range");
      return tensors[index];
    }

    const math::clFTensor &getTensor(size_t index) const {
      if (index > tensors.size()) throw std::out_of_range("Tensor index out of range");
      return tensors[index];
    }

    std::vector<math::clFTensor> &getTensors() { return tensors; }
    const std::vector<math::clFTensor> &getTensors() const { return tensors; }

    using TensorIterator = std::vector<math::clFTensor>::iterator;
    using TensorConstIterator = std::vector<math::clFTensor>::const_iterator;

    TensorIterator beginTensor() { return tensors.begin(); }

    TensorIterator endTensor() { return tensors.end(); }

    TensorConstIterator beginTensor() const { return tensors.begin(); }

    TensorConstIterator endTensor() const { return tensors.end(); }

  private:
    std::vector<Sample> samples;
    std::vector<math::clFTensor> tensors;
  };

}   // namespace control
