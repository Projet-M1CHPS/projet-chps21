#pragma once


#include "Matrix.hpp"
#include <utility>


namespace cnnet {

  class Filter {
  public:
    Filter(const size_t rows, const size_t cols);
    Filter(const std::pair<size_t, size_t>& sizeFilter);
    Filter(const Filter &other) = delete;
    Filter const &operator=(const Filter &other) = delete;

    ~Filter() = default;

    void randomizeFilter();

    const size_t getRows() const { return filter.getRows(); }
    const size_t getCols() const { return filter.getCols(); }

  private:
    math::FloatMatrix filter;
  };

}   // namespace cnnet