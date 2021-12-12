#pragma once

#include "Matrix.hpp"
#include <vector>


namespace nnet {

  template<typename T = float>
  class BackpropStorage {
  public:
    explicit BackpropStorage(std::vector<math::Matrix<T>> &w) : weights(&w){};

    BackpropStorage(BackpropStorage const &other) = delete;
    BackpropStorage(BackpropStorage &&other) noexcept = default;

    BackpropStorage &operator=(BackpropStorage const &other) = delete;
    BackpropStorage &operator=(BackpropStorage &&other) = default;

    ~BackpropStorage() = default;

    math::Matrix<T> &getWeights(size_t i) { return weights->at(i); }
    math::Matrix<T> const &getWeights(size_t i) const { return weights->at(i); }

    math::Matrix<T> &getWeights() { return weights->at(index); }
    math::Matrix<T> const &getWeights() const { return weights->at(index); }

    const long getIndex() const { return index; }
    void setIndex(const long i) { index = i; }

    math::Matrix<T> &getGradient() { return gradient; }
    const math::Matrix<T> &getGradient() const { return gradient; }

    math::Matrix<T> &getError() { return current_error; }
    const math::Matrix<T> &getError() const { return current_error; }

  private:
    long index = 0;

    std::vector<math::Matrix<T>> *weights;
    math::Matrix<T> gradient;
    math::Matrix<T> current_error;
  };

}   // namespace nnet