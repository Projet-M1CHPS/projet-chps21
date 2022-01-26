#pragma once

#include "Matrix.hpp"
#include <vector>


namespace nnet {

  class BackpropStorage {
  public:
    explicit BackpropStorage() = default;
    explicit BackpropStorage(std::vector<math::FloatMatrix> &w) : weights(&w){};

    BackpropStorage(BackpropStorage const &other) = delete;
    BackpropStorage(BackpropStorage &&other) noexcept = default;

    BackpropStorage &operator=(BackpropStorage const &other) = delete;
    BackpropStorage &operator=(BackpropStorage &&other) = default;

    ~BackpropStorage() = default;

    math::FloatMatrix &getWeights(size_t i) { return weights->at(i); }
    math::FloatMatrix const &getWeights(size_t i) const { return weights->at(i); }

    math::FloatMatrix &getWeights() { return weights->at(index); }
    math::FloatMatrix const &getWeights() const { return weights->at(index); }

    const long getIndex() const { return index; }
    void setIndex(const long i) { index = i; }

    math::FloatMatrix &getGradient() { return gradient; }
    const math::FloatMatrix &getGradient() const { return gradient; }

    math::FloatMatrix &getError() { return current_error; }
    const math::FloatMatrix &getError() const { return current_error; }

  private:
    long index = 0;

    std::vector<math::FloatMatrix> *weights;
    math::FloatMatrix gradient;
    math::FloatMatrix  current_error;
  };

}   // namespace nnet