#pragma once
#include "clUtils/clFMatrix.hpp"
#include "clUtils/clFTensor.hpp"
#include <filesystem>
#include <iostream>

namespace control {

  class Sample {
  public:
    Sample(size_t id, std::filesystem::path const &path, math::clFMatrix data);
    Sample(size_t id, std::filesystem::path const &path, math::clFMatrix &&data);

    size_t getId() const;
    size_t getClass() const;
    void setClass(size_t class_id);

    const std::filesystem::path &getPath() const;
    math::clFMatrix &getData();

  private:
    size_t id;
    size_t class_id;
    std::filesystem::path path;
    math::clFMatrix data;
  };

  class InputSet {
  public:
    static std::unique_ptr<InputSet> load(std::filesystem::path const &path);
    static std::unique_ptr<InputSet> loadLabelled(std::filesystem::path const &path);

    void append(std::vector<Sample> &&sample, math::clFTensor &&tensor);

    Sample &operator[](size_t index);
    const Sample &operator[](size_t index) const;

    using SampleIterator = std::vector<Sample>::iterator;
    using SampleConstIterator = std::vector<Sample>::const_iterator;

    SampleIterator begin();
    SampleIterator end();

    SampleConstIterator begin() const;
    SampleConstIterator end() const;

    using TensorIterator = std::vector<math::clFTensor>::iterator;
    using TensorConstIterator = std::vector<math::clFTensor>::const_iterator;

    size_t getTensorCount() const;

    math::clFTensor &getTensor(size_t index);
    const math::clFTensor &getTensor(size_t index) const;

    TensorIterator beginTensor();
    TensorIterator endTensor();

    TensorConstIterator beginTensor() const;
    TensorConstIterator endTensor() const;

  private:
    std::vector<Sample> samples;
    std::vector<math::clFTensor> tensors;
  };

}   // namespace control
