#pragma once

#include "Matrix.hpp"
#include <vector>


namespace nnet {

  template<typename T>
  class BackpropStorage {
  public:
    BackpropStorage(std::vector<math::Matrix<T>> &w) : weights(w){};
    ~BackpropStorage() = default;

    math::Matrix<T> &getWeights() const { return weights[index]; }

    const long getIndex() const { return index; }
    void setIndex(const long i) { index = i; }

    const math::Matrix<T> &getGradient() const { return gradient; }
    void setGradient(const math::Matrix<T> &g) { gradient = g; }

    const math::Matrix<T> &getError() const { return current_error; }
    void setError(const math::Matrix<T> &e) { current_error = e; }

  public:
    std::vector<math::Matrix<T>> &weights;
    long index = 0;
    math::Matrix<T> gradient;
    math::Matrix<T> current_error;
  };

}   // namespace nnet