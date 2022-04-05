#include "clFTensor.hpp"

namespace math {

  clFTensor::clFTensor(size_t x, size_t y, size_t z) : x_dim(x), y_dim(y), z_dim(z) {
    size_t size = x * y * z;
    data = cl::Buffer(CL_MEM_READ_WRITE, size * sizeof(float));
  }

  clFTensor clFTensor::deepCopy(cl::CommandQueue &queue, bool blocking) const {
    clFTensor res(x_dim, y_dim, z_dim);
    size_t size = x_dim * y_dim;
    queue.enqueueCopyBuffer(res.data, data, offset * size, 0, data.getInfo<CL_MEM_SIZE>());
    return res;
  }

  clFMatrix clFTensor::getMatrix(size_t z) {
    if (z > z_dim) { throw std::out_of_range("z index out of range"); }

    size_t size = x_dim * y_dim;
    size_t mat_offset = (z + offset) * size;
    return clFMatrix::fromSubbuffer(data, x_dim, y_dim, mat_offset);
  }

  clFMatrix clFTensor::getMatrix(size_t z) const {
    if (z > z_dim) { throw std::out_of_range("z index out of range"); }

    size_t size = x_dim * y_dim;
    size_t mat_offset = (z + offset) * size;
    return clFMatrix::fromSubbuffer(data, x_dim, y_dim, mat_offset);
  }

  std::vector<clFMatrix> clFTensor::getMatrices() {
    std::vector<clFMatrix> matrices;
    for (size_t i = 0; i < z_dim; i++) { matrices.push_back(getMatrix(i)); }
    return matrices;
  }

}   // namespace math
