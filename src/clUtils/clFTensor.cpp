#include "clFTensor.hpp"

namespace math {

  clFTensor::clFTensor(size_t x, size_t y, size_t z) : x_dim(x), y_dim(y), z_dim(z) {
    size_t size = x * y * z;
    data = cl::Buffer(CL_MEM_READ_WRITE, size * sizeof(float));
  }

  clFTensor &clFTensor::copy(const clFTensor &other, cl::CommandQueue &queue, bool blocking) {
    size_t size = x_dim * y_dim * z_dim;
    size_t other_size = other.x_dim * other.y_dim * other.z_dim;

    if (size != other_size) {
      if (offset != 0)
        throw std::runtime_error("clFTensor::copy: Cannot copy a tensor with a different size, "
                                 "when the destination is a sub-tensor");
      if (other_size == 0) data = cl::Buffer();
      else
        data = cl::Buffer(CL_MEM_READ_WRITE, other_size * sizeof(float));
    }

    x_dim = other.x_dim;
    y_dim = other.y_dim;
    z_dim = other.z_dim;

    // We need to return if the size is 0 else OpenCL will throw
    if (other_size != 0) {
      cl::Event evt;
      size_t mat_size = x_dim * y_dim * sizeof(float);
      queue.enqueueCopyBuffer(other.data, data, other.offset * mat_size, offset * mat_size,
                              other_size * sizeof(float), nullptr, &evt);
      if (blocking) evt.wait();
    }
    return *this;
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

  std::vector<clFTensor> clFTensor::shallowSplit(size_t ndiv) const {
    std::vector<clFTensor> parts;
    size_t z_dim_part = z_dim / ndiv;
    size_t z_dim_remainder = z_dim % ndiv;

    size_t local_offset = 0;
    for (size_t i = 0; i < ndiv; i++) {
      parts.push_back(shallowCopy());
      parts.back().z_dim = z_dim_part;
      parts.back().offset = local_offset;
      if (i < z_dim_remainder) { parts.back().z_dim++; }
      local_offset += parts.back().z_dim;
    }
    return parts;
  }

  clFTensor clFTensor::sub(float factor, const clFTensor &other, cl::CommandQueue &queue,
                           bool blocking) const {
    if (x_dim != other.x_dim || y_dim != other.y_dim || z_dim != other.z_dim) {
      throw std::runtime_error("clFTensor::sub: Cannot subtract two tensors with different sizes");
    }

    clFTensor result(x_dim, y_dim, z_dim);
    result.copy(*this, queue, false);
    size_t size = x_dim * y_dim;

    cl::Event evt;
    clblast::Axpy<float>(size * z_dim, -factor, other.data(), other.offset, 1, result.data(),
                         result.offset, 1, &queue(), &evt());
    if (blocking) evt.wait();
    return result;
  }

  clFTensor clFTensor::batchedGemm(float alpha, bool transpose_a, const clFMatrix &A,
                                   bool transpose_b, const clFTensor &B, cl::CommandQueue &queue,
                                   bool blocking) {
    const size_t A_rows = A.getRows(), A_cols = A.getCols(), B_rows = B.getX(), B_cols = B.getY();

    if ((transpose_a ? A_rows : A_cols) != (transpose_b ? B_cols : B_rows)) {
      throw std::invalid_argument("Matrix size do not match");
    }

    auto ta = transpose_a ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    auto tb = transpose_b ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    size_t m = (transpose_a ? A_cols : A_rows);
    size_t n = (transpose_b ? B_rows : B_cols);
    size_t k = (transpose_a ? A_rows : A_cols);

    clFTensor res((transpose_a ? A_cols : A_rows), (transpose_b ? B_rows : B_cols), B.z_dim);
    cl::Event evt;

    std::vector<float> alphas(B.z_dim, alpha);
    std::vector<float> betas(B.z_dim, 0.0f);
    std::vector<size_t> a_offset(B.z_dim, A.getOffset());

    std::vector<size_t> b_offsets(B.z_dim);
    for (size_t i = 0; i < B.z_dim; i++) {
      size_t mat_offset = (i + B.offset) * (B.x_dim * B.y_dim);
      b_offsets[i] = mat_offset;
    }

    std::vector<size_t> c_offset(B.z_dim);
    for (size_t i = 0; i < B.z_dim; i++) {
      size_t mat_offset = (i + res.offset) * (res.x_dim * res.y_dim);
      c_offset[i] = mat_offset;
    }

    clblast::GemmBatched<float>(clblast::Layout::kRowMajor, ta, tb, m, n, k, alphas.data(),
                                A.getBuffer()(), a_offset.data(), A_cols, B.data(),
                                b_offsets.data(), B_cols, betas.data(), res.data(), c_offset.data(),
                                res.getY(), B.z_dim, &queue(), &evt());
    if (blocking) evt.wait();

    return res;
  }

  clFTensor clFTensor::batchedGemm(float alpha, bool transpose_a, const clFMatrix &A,
                                   bool transpose_b, const clFTensor &B, float beta,
                                   const clFMatrix &C, cl::CommandQueue &queue, bool blocking) {
    const size_t A_rows = A.getRows(), A_cols = A.getCols(), B_rows = B.getX(), B_cols = B.getY(),
                 C_rows = C.getRows(), C_cols = C.getCols();

    if ((transpose_a ? A_rows : A_cols) != (transpose_b ? B_cols : B_rows)) {
      throw std::invalid_argument("Matrix size do not match");
    }

    auto ta = transpose_a ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    auto tb = transpose_b ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    size_t m = (transpose_a ? A_cols : A_rows);
    size_t n = (transpose_b ? B_rows : B_cols);
    size_t k = (transpose_a ? A_rows : A_cols);

    clFTensor res(C_rows, C_cols, B.z_dim);
    for (size_t i = 0; i < res.z_dim; i++) { res.getMatrix(i).copy(C, queue, false); }
    cl::Event evt;

    std::vector<float> alphas(B.z_dim, alpha);
    std::vector<float> betas(B.z_dim, 1.0f);
    std::vector<size_t> a_offset(B.z_dim, A.getOffset());

    std::vector<size_t> b_offsets(B.z_dim);
    for (size_t i = 0; i < B.z_dim; i++) {
      size_t mat_offset = (i + B.offset) * (B.x_dim * B.y_dim);
      b_offsets[i] = mat_offset;
    }

    std::vector<size_t> c_offset(B.z_dim);
    for (size_t i = 0; i < B.z_dim; i++) {
      size_t mat_offset = (i + res.offset) * (res.x_dim * res.y_dim);
      c_offset[i] = mat_offset;
    }

    queue.finish();

    FloatMatrix tmp = A.toFloatMatrix();
    tmp = B.getMatrix(0).toFloatMatrix();
    tmp = res.getMatrix(0).toFloatMatrix();

    clblast::GemmBatched<float>(clblast::Layout::kRowMajor, ta, tb, m, n, k, alphas.data(),
                                A.getBuffer()(), a_offset.data(), A_cols, B.data(),
                                b_offsets.data(), B_cols, betas.data(), res.data(), c_offset.data(),
                                C_cols, B.z_dim, &queue(), &evt());
    queue.finish();
    tmp = res.getMatrix(0).toFloatMatrix();
    if (blocking) evt.wait();

    return res;
  }


  clFTensor clFTensor::batchedGemm(float alpha, bool transpose_a, const clFTensor &A,
                                   bool transpose_b, const clFTensor &B, cl::CommandQueue &queue,
                                   bool blocking) {
    const size_t A_rows = A.getX(), A_cols = A.getY(), B_rows = B.getX(), B_cols = B.getY();

    if ((transpose_a ? A_rows : A_cols) != (transpose_b ? B_cols : B_rows)) {
      throw std::invalid_argument("Matrix size do not match");
    }

    auto ta = transpose_a ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    auto tb = transpose_b ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    size_t m = (transpose_a ? A_cols : A_rows);
    size_t n = (transpose_b ? B_rows : B_cols);
    size_t k = (transpose_a ? A_rows : A_cols);

    clFTensor res((transpose_a ? A_cols : A_rows), (transpose_b ? B_rows : B_cols), B.z_dim);
    cl::Event evt;

    std::vector<float> alphas(A.z_dim, alpha);
    std::vector<float> betas(res.z_dim, 0.0f);
    std::vector<size_t> a_offset(B.z_dim);
    for (size_t i = 0; i < A.z_dim; i++) {
      size_t mat_offset = (i + A.offset) * (A.x_dim * A.y_dim);
      a_offset[i] = mat_offset;
    }

    std::vector<size_t> b_offsets(B.z_dim);
    for (size_t i = 0; i < B.z_dim; i++) {
      size_t mat_offset = (i + B.offset) * (B.x_dim * B.y_dim);
      b_offsets[i] = mat_offset;
    }

    std::vector<size_t> c_offset(res.z_dim);
    for (size_t i = 0; i < res.z_dim; i++) {
      size_t mat_offset = (i + res.offset) * (res.x_dim * res.y_dim);
      c_offset[i] = mat_offset;
    }

    clblast::GemmBatched<float>(clblast::Layout::kRowMajor, ta, tb, m, n, k, alphas.data(),
                                A.getBuffer()(), a_offset.data(), A_cols, B.data(),
                                b_offsets.data(), B_cols, betas.data(), res.data(), c_offset.data(),
                                res.getY(), B.z_dim, &queue(), &evt());
    if (blocking) evt.wait();

    return res;
  }

  clFMatrix clFTensor::meanSumCollapse(cl::CommandQueue &queue, cl::Event &event,
                                       bool blocking) const {
    clFMatrix result(x_dim, y_dim);
    result.fill(0.0f, queue, false);
    if (z_dim == 0) { return result; }

    float factor = 1.0f / z_dim;
    for (size_t i = 0; i < z_dim; i++) {
      clFMatrix mat = getMatrix(i);
      clblast::Axpy<float>(x_dim * y_dim, factor, mat.getBuffer()(), mat.getOffset(), 1,
                           result.getBuffer()(), result.getOffset(), 1, &queue(), &event());
    }
    if (blocking) event.wait();
    return result;
  }


  clFTensor &clFTensor::iphadamard(const clFTensor &other, cl::CommandQueue &queue, bool blocking) {
    if (x_dim != other.x_dim || y_dim != other.y_dim || z_dim != other.z_dim) {
      throw std::runtime_error("clFTensor::sub: Cannot subtract two tensors with different sizes");
    }

    size_t size = x_dim * y_dim;

    cl::Event evt;

    clblast::Had<float>(size * z_dim, 1.0f, other.data(), other.offset * size, 1, data(),
                        offset * size, 1, 0.0f, data(), offset * size, 1, &queue(), &evt());
    if (blocking) evt.wait();
    return *this;
  }

}   // namespace math
