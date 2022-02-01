#include "ConvolutionLayer.hpp"

namespace cnnet {


  ConvolutionLayer::ConvolutionLayer(const size_t rowsFiltre, const size_t colsFiltre,
                                     const size_t stride, const size_t padding)
      : filter(rowsFiltre, colsFiltre), stride(stride), padding(padding) {}


  ConvolutionLayer::ConvolutionLayer(std::pair<size_t, size_t> sizeFilter, const size_t stride,
                                     const size_t padding)
      : filter(sizeFilter), stride(stride), padding(padding) {}


  void ConvolutionLayer::compute(const FloatMatrix &input, FloatMatrix &output) {
    const size_t max = ((input.getRows() - filter.getRows() + 2 * padding) / stride) + 1; //* nombre de layer;
    
    const FloatMatrix& matFiltre = filter.getMatrix();

    int rowsPos = -padding;
    int colsPos = -padding;
    
    for (size_t i = 0; i < max; i++)
    {
      for (size_t j = 0; j < max; j++)
      {
        float sum = 0.f;
        //std::cout << "sum = 0" << std::endl;
        for (size_t k = 0; k < matFiltre.getRows(); k++)
        {
          for (size_t l = 0; l < matFiltre.getCols(); l++)
          {
            sum += input(k + rowsPos, l + colsPos) * matFiltre(k, l);
          }
        }
        output(i, j) = sum;
        colsPos += stride;
      }
      rowsPos += stride;
      colsPos = -padding;
    }
  }
}   // namespace cnnet