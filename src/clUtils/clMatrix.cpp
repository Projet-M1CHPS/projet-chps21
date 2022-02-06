#include "clMatrix.hpp"
#include "Matrix.hpp"

namespace math {
  Matrix<float> fetchClMatrix(const clMatrix &matrix, const cl::CommandQueue &queue) {
    math::Matrix<float> res(matrix.getRows(), matrix.getCols());
    queue.enqueueReadBuffer(matrix.getBuffer(), CL_TRUE, 0,
                            matrix.getRows() * matrix.getCols() * sizeof(float), res.getData());
    return res;
  }
}   // namespace math