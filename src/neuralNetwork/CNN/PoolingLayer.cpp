#include "PoolingLayer.hpp"


namespace cnnet {

  PoolingLayer::PoolingLayer(const size_t stride, const size_t padding)
      : stride(stride), padding(padding) {}


  void PoolingLayer::compute(const FloatMatrix &input, const FloatMatrix &output) {
    // on deplace le kernel et on fait la fonction
    poolingMethode();
  }


  MaxPoolingLayer::MaxPoolingLayer(const size_t stride, const size_t padding)
      : PoolingLayer(stride, padding) {}

  const float poolingMethode()
  {
    return 0.f;
  }


  AvgPoolingLayer::AvgPoolingLayer(const size_t stride, const size_t padding)
      : PoolingLayer(stride, padding) {}

  const float AvgPoolingLayer::poolingMethode()
  {
    return 0.f;
  }


}   // namespace cnnet