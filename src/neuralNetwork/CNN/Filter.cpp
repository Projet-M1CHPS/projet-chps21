#include "Filter.hpp"


namespace cnnet {


  Filter::Filter(const size_t rows, const size_t cols) : filter(rows, cols) {}

  Filter::Filter(const std::pair<size_t, size_t> &sizeFilter)
      : filter(sizeFilter.first, sizeFilter.second) {}

  void Filter::randomizeFilter() { randomize(filter, 0.f, 1.f); }

}   // namespace cnnet