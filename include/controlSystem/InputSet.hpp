#pragma once
#include "math/clFMatrix.hpp"
#include "math/clFTensor.hpp"
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

    long getClass() const { return class_id; }

    void setClass(long new_id) { class_id = new_id; }

    /**
     * @brief Returns the opencl buffer containing the data of this sample
     * Note that the data returned may be part of a larger tensor
     * @return
     */
    math::clFMatrix &getData() { return data; }
    const math::clFMatrix &getData() const { return data; }

  private:
    size_t id;
    long class_id;
    math::clFMatrix data;
  };

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
    InputSet(size_t input_width, size_t input_height)
        : input_width(input_width), input_height(input_height) {}

    InputSet(const InputSet &) = delete;
    InputSet &operator=(const InputSet &) = delete;

    InputSet(InputSet &&other) noexcept { *this = std::move(other); }

    InputSet &operator=(InputSet &&other) noexcept;

    /**
     * @brief Append the given samples and their associated tensor to the input set. Samples are
     * added at the end of the set.
     *
     * @param tensor The tensor containing the samples
     * @param ids A vector containing the ids of the samples to append
     * @param class_ids A vector containing the class ids of the samples to append. If the size of
     * this vector is smaller than the size of the ids vector, the class id is ignored. This allow
     * one to create an InputSet where the classes are not known
     */
    void append(math::clFTensor &&tensor, const std::vector<size_t> &ids,
                const std::vector<long> &class_ids = {});

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

    void split(size_t nb_sets, std::vector<InputSet> &sub_sets) const;

    size_t getSize() const {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return samples.size();
    }

    size_t getInputWidth() const { return input_width; }
    size_t getInputHeight() const { return input_height; }

    Sample &operator[](size_t index) { return samples[index]; }
    const Sample &operator[](size_t index) const { return samples[index]; }

    using SampleIterator = std::vector<Sample>::iterator;
    using ConstSampleIterator = std::vector<Sample>::const_iterator;

    SampleIterator begin() {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return samples.begin();
    }

    ConstSampleIterator begin() const {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return samples.begin();
    }

    SampleIterator end() {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return samples.end();
    }

    ConstSampleIterator end() const {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return samples.end();
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

    /**
     * @brief Return the class id of the sample at the given index
     * @param global_index The index of the sample in the input set
     * @return The class id of the sample, -1 if it is undefined
     * Throw on out of range
     */
    long getClassOf(size_t global_index) const {
      std::shared_lock lock(mutex);
      if (global_index > samples.size()) throw std::out_of_range("Sample index out of range");
      return samples[global_index].getClass();
    }

    size_t getClassCount() const {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return class_names.size();
    }


    using TensorIterator = std::vector<math::clFTensor>::iterator;
    using TensorConstIterator = std::vector<math::clFTensor>::const_iterator;

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
    std::vector<std::string> &getClasses() {
      std::shared_lock lock(mutex);
      return class_names;
    }

    /**
     * @brief Return the class id of the sample at the given index
     * @param global_index The index of the sample in the input set
     * @return The class id of the sample, -1 if it is undefined
     * Throw on out of range
     */
    const std::vector<std::string> &getClasses() const {
      std::shared_lock lock(mutex);
      return class_names;
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

    /**
     * @brief Return the id for a specific sample in the input set
     * @return The id of the sample
     */
    size_t getSampleId(size_t index) const {
      std::scoped_lock lock(mutex);
      return samples.at(index).getId();
    }

    /**
     * @brief Return the id for each of the samples in the input set
     * @return The id of each of the samples
     */
    std::vector<size_t> getSamplesIds() const {
      std::scoped_lock lock(mutex);
      std::vector<size_t> ids;
      std::for_each(samples.cbegin(), samples.cend(),
                    [&ids](const Sample &sample) { ids.push_back(sample.getId()); });
      return ids;
    }

    /**
     * @brief Return the class id for each of the samples in the input set
     * @return The class id of each of the samples
     */
    std::vector<long> getSamplesClassIds() const {
      std::scoped_lock lock(mutex);
      std::vector<long> class_ids;
      std::for_each(samples.cbegin(), samples.cend(),
                    [&class_ids](const Sample &sample) { class_ids.push_back(sample.getClass()); });
      return class_ids;
    }

  private:
    // Split the tensors into nb_sets parts.
    std::vector<std::vector<math::clFTensor>> splitTensors(size_t nb_sets) const;

    // Split samples into nb_sets parts.
    std::vector<std::vector<Sample>> splitSamples(size_t nb_sets) const;

    // Split class names into nb_sets parts.
    std::vector<std::vector<std::string>> splitClassNames(size_t nb_sets) const;

    // rows and cols dimension of the matrices/tensors
    size_t input_width = 0, input_height = 0;

    // Store each sample in a vector.
    // This induces some data duplication as each sample store the associated matrix, which are also
    // included in the tensors vector. However, doing so allows returning iterators on the samples,
    // which is really practical.
    std::vector<Sample> samples;

    // Store the tensor separately to avoid copying the data
    std::vector<math::clFTensor> tensors;
    std::vector<std::string> class_names;

    mutable std::shared_mutex mutex;
  };
}   // namespace control
