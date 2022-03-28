#include "Filter.hpp"


namespace cnnet {


  Filter::Filter(const size_t rows, const size_t cols) : filter(rows, cols) {}

  Filter::Filter(const std::pair<size_t, size_t> &sizeFilter)
      : filter(sizeFilter.first, sizeFilter.second) {
    filter(0, 0) = 2;
    filter(0, 1) = 1;
    filter(1, 0) = 0.5;
    filter(1, 1) = 1.5;

    // filter(0, 0) = 0.5;
    // filter(0, 1) = 0.5;
    // filter(0, 2) = 1;
    // filter(1, 0) = 1;
    // filter(1, 1) = 2;
    // filter(1, 2) = 0.5;
    // filter(2, 0) = 1;
    // filter(2, 1) = 2;
    // filter(2, 2) = 0.5;
  }

  void Filter::randomize(const float min, const float max) {
    math::randomize<float>(filter, min, max);
  }

}   // namespace cnnet