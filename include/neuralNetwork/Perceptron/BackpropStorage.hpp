#pragma once

#include "Matrix.hpp"
#include "clUtils/clFMatrix.hpp"
#include <vector>


namespace nnet {

  class BackpropStorage {
  public:
    explicit BackpropStorage() = default;
    explicit BackpropStorage(std::vector<math::clFMatrix> &w) : weights(&w){};

    BackpropStorage(BackpropStorage const &other) = delete;
    BackpropStorage(BackpropStorage &&other) noexcept = default;

    BackpropStorage &operator=(BackpropStorage const &other) = delete;
    BackpropStorage &operator=(BackpropStorage &&other) = default;

    ~BackpropStorage() = default;

    math::clFMatrix &getWeights(size_t i) { return weights->at(i); }
    math::clFMatrix const &getWeights(size_t i) const { return weights->at(i); }

    math::clFMatrix &getWeights() { return weights->at(index); }
    math::clFMatrix const &getWeights() const { return weights->at(index); }

    const long getIndex() const { return index; }
    void setIndex(const long i) { index = i; }

    math::clFMatrix &getGradient() { return gradient; }
    const math::clFMatrix &getGradient() const { return gradient; }

    math::clFMatrix &getError() { return current_error; }
    const math::clFMatrix &getError() const { return current_error; }

  private:
    long index = 0;

    std::vector<math::clFMatrix> *weights;
    math::clFMatrix gradient;
    math::clFMatrix current_error;
  };

}   // namespace nnet