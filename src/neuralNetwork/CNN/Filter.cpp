#include "Filter.hpp"


namespace cnnet {


  Filter::Filter(const size_t rows, const size_t cols) : filter(rows, cols) {
    randomize(filter, 0.f, 1.f);
  }

  Filter::Filter(const std::pair<size_t, size_t> &sizeFilter)
      : filter(sizeFilter.first, sizeFilter.second) {
    filter(0, 0) = 2;
    filter(0, 1) = 1;
    filter(1, 0) = 0.5;
    filter(1, 1) = 1.5;
  }


}   // namespace cnnet