#pragma once


#include "Matrix.hpp"
#include <utility>


namespace cnnet {

  using namespace math;
  class Filter {
  public:
    Filter(const size_t rows, const size_t cols);
    Filter(const std::pair<size_t, size_t> &sizeFilter);
    Filter(const Filter &other) = default;
    Filter(Filter &&other) = default;
    Filter const &operator=(const Filter &other) = delete;

    ~Filter() = default;

    const size_t getRows() const { return filter.getRows(); }
    const size_t getCols() const { return filter.getCols(); }
    const FloatMatrix &getMatrix() const { return filter; }
    FloatMatrix &getMatrix() { return filter; }

    void randomize(const float min, const float max);

  private:
    FloatMatrix filter;
  };

}   // namespace cnnet