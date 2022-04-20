#include "clFTensor.hpp"

namespace math {

  namespace {
    // clblast batched operations takes a vector of offsets
    // Define a few helper functions to make the code more readable
    std::vector<size_t> makeOffsetVector(const clFTensor &tensor) {
      std::vector<size_t> res;
      res.reserve(tensor.getDepth());
      for (size_t i = 0; i < tensor.getDepth(); i++) { res[i] = tensor.getOffsetOf(i); }
      return res;
    }

    std::vector<size_t> makeNullOffsetVector(const clFMatrix &matrix, size_t count) {
      std::vector<size_t> res(count, matrix.getOffset());
      return res;
    }
  }   // namespace

  std::ostream &operator<<(std::ostream &os, const clFTensor &t) {
    if (t.is_view) std::cout << "(View) ";
    std::cout << "clFTensor(" << t.rows << ", " << t.cols << ", " << t.depth << "):" << std::endl;
    for (size_t i = 0; i < t.depth; i++) { std::cout << t[i] << std::endl; }
    return os;
  }

  clFTensor::clFTensor(size_t width, size_t height, size_t depth)
      : rows(width), cols(height), depth(depth), is_view(false) {
    data = cl::Buffer(CL_MEM_READ_WRITE, sizeInBytes());
  }

  clFTensor &clFTensor::copy(const clFTensor &other, cl::CommandQueue &queue, bool blocking) {
    if (size() != other.size()) {
      if (is_view)
        throw std::runtime_error("clFTensor::copy: Tried to copy a tensor inside a smaller view. "
                                 "This would break the view");
      if (other.size() == 0) data = cl::Buffer();
      else
        data = cl::Buffer(CL_MEM_READ_WRITE, other.sizeInBytes());
    }

    rows = other.rows;
    cols = other.cols;
    depth = other.depth;

    // We need to return if the size is 0 else OpenCL will throw
    if (size() != 0) {
      cl::Event evt;
      queue.enqueueCopyBuffer(other.data, data, other.getOffsetInBytes(), getOffsetInBytes(),
                              sizeInBytes(), nullptr, &evt);
      if (blocking) evt.wait();
    }
    return *this;
  }

  clFMatrix clFTensor::getMatrix(size_t z) {
    if (z > depth) { throw std::out_of_range("clFTensor::getMatrix: z index out of range"); }

    return {data, rows, cols, getOffsetOf(z)};
  }

  clFMatrix clFTensor::getMatrix(size_t z) const {
    if (z > depth) { throw std::out_of_range("clFTensor::getMatrix: z index out of range"); }

    return {data, rows, cols, getOffsetOf(z)};
  }

  std::vector<clFMatrix> clFTensor::getMatrices() {
    std::vector<clFMatrix> matrices;
    matrices.reserve(depth);
    for (size_t i = 0; i < depth; i++) { matrices.emplace_back((*this)[i]); }
    return matrices;
  }

  [[nodiscard]] std::vector<clFMatrix> clFTensor::getMatrices() const {
    std::vector<clFMatrix> matrices;
    matrices.reserve(depth);
    for (size_t i = 0; i < depth; i++) { matrices.emplace_back((*this)[i]); }
    return matrices;
  }

  clFTensor clFTensor::shallowCopy() const {
    clFTensor copy;
    copy.rows = rows;
    copy.cols = cols;
    copy.depth = depth;
    copy.data = data;
    copy.offset = offset;
    return copy;
  }

  std::vector<clFTensor> clFTensor::slice(size_t ndiv) const {
    std::vector<clFTensor> parts;

    size_t z_dim_part = depth / ndiv;
    size_t z_dim_remainder = depth % ndiv;

    size_t local_offset = 0;
    for (size_t i = 0; i < ndiv; i++) {
      clFTensor part = shallowCopy();
      part.depth = i < z_dim_remainder ? z_dim_part + 1 : z_dim_part;
      part.offset = local_offset;
      part.is_view = true;
      local_offset += part.depth;
      parts.push_back(std::move(part));
    }
    return parts;
  }

  clFTensor clFTensor::slice(size_t begin, size_t end) const {
    if (begin > depth or begin > end or end > depth) {
      throw std::out_of_range("clFTensor::slice: begin or end index out of range");
    }
    clFTensor slice = shallowCopy();
    slice.depth = end - begin;
    slice.offset = begin;
    slice.is_view = true;
    return slice;
  }

  clFTensor clFTensor::flatten() const {
    clFTensor res;
    res.data = data;
    res.rows = rows * cols;
    // If the tensor is of size (5, 0, 10)
    // Then the flattened tensor will have size (0, 0, 10)
    // so the cols is not always 1
    res.cols = cols > 0 ? 1 : 0;
    res.depth = depth;
    res.offset = offset;
    return res;
  }

  void clFTensor::reshape(size_t new_rows, size_t new_cols, size_t new_depth) {
    if (rows * cols * depth != new_rows * new_cols * new_depth)
      throw std::invalid_argument("clFTensor::reshape: New size does not match old size");

    rows = new_rows;
    cols = new_cols;
    depth = new_depth;
  }

  clFTensor clFTensor::sub(float factor, const clFTensor &other, cl::CommandQueue &queue,
                           bool blocking) const {
    if (rows != other.rows || cols != other.cols || depth != other.depth) {
      throw std::runtime_error("clFTensor::sub: Cannot subtract two tensors with different sizes");
    }

    clFTensor result(rows, cols, depth);
    result.copy(*this, queue, false);

    cl::Event evt;
    clblast::Axpy<float>(size(), -factor, other.data(), other.getOffsetInFloats(), 1, result.data(),
                         result.getOffsetInFloats(), 1, &queue(), &evt());
    if (blocking) evt.wait();
    return result;
  }

  void clFTensor::ipadd(float factor, const clFTensor &other, cl::CommandQueue &queue,
                        bool blocking) {
    if (rows != other.rows || cols != other.cols || depth != other.depth) {
      throw std::runtime_error("clFTensor::sub: Cannot subtract two tensors with different sizes");
    }

    cl::Event evt;
    clblast::Axpy<float>(size(), factor, other.data(), other.getOffsetInFloats(), 1, data(),
                         getOffsetInFloats(), 1, &queue(), &evt());
    if (blocking) evt.wait();
  }

  clFTensor clFTensor::batchedGemm(float alpha, bool transpose_a, const clFMatrix &A,
                                   bool transpose_b, const clFTensor &B, cl::CommandQueue &queue,
                                   bool blocking) {
    const size_t A_rows = A.getRows(), A_cols = A.getCols(), B_rows = B.getRows(),
                 B_cols = B.getCols();

    if ((transpose_a ? A_rows : A_cols) != (transpose_b ? B_cols : B_rows))
      throw std::invalid_argument("clFTensor::batchedGemm: Matrix size do not match");

    auto ta = transpose_a ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    auto tb = transpose_b ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    size_t m = (transpose_a ? A_cols : A_rows);
    size_t n = (transpose_b ? B_rows : B_cols);
    size_t k = (transpose_a ? A_rows : A_cols);

    clFTensor res((transpose_a ? A_cols : A_rows), (transpose_b ? B_rows : B_cols), B.depth);
    res.fill(0, queue, blocking);
    cl::Event evt;

    std::vector<float> alphas(B.depth, alpha);
    std::vector<float> betas(B.depth, 0.0f);

    auto a_offset = makeNullOffsetVector(A, res.depth);
    auto b_offsets = makeOffsetVector(B);
    auto c_offset = makeOffsetVector(res);

    clblast::GemmBatched<float>(clblast::Layout::kRowMajor, ta, tb, m, n, k, alphas.data(),
                                A.getBuffer()(), a_offset.data(), A_cols, B.data(),
                                b_offsets.data(), B_cols, betas.data(), res.data(), c_offset.data(),
                                res.getCols(), res.depth, &queue(), &evt());
    if (blocking) evt.wait();

    return res;
  }

  clFTensor clFTensor::batchedGemm(float alpha, bool transpose_a, const clFMatrix &A,
                                   bool transpose_b, const clFTensor &B, float beta,
                                   const clFMatrix &C, cl::CommandQueue &queue, bool blocking) {
    const size_t A_rows = A.getRows(), A_cols = A.getCols(), B_rows = B.getRows(),
                 B_cols = B.getCols(), C_rows = C.getRows(), C_cols = C.getCols();

    if ((transpose_a ? A_rows : A_cols) != (transpose_b ? B_cols : B_rows))
      throw std::invalid_argument("clFTensor::batchedGemm: Matrix size do not match");

    auto ta = transpose_a ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    auto tb = transpose_b ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    size_t m = (transpose_a ? A_cols : A_rows);
    size_t n = (transpose_b ? B_rows : B_cols);
    size_t k = (transpose_a ? A_rows : A_cols);

    // Where _t denotes a tensor
    // R_t = alpha * A * B_t + beta * C
    // Since gemm will use R_t as the output, we must fill C to R_t
    clFTensor res(C_rows, C_cols, B.depth);
    res.fill(0.0f, queue, false);

    std::vector<float> alphas(B.depth, alpha);
    std::vector<float> betas(B.depth, 1.0f);
    std::vector<size_t> a_offset(B.depth, A.getOffset());

    auto b_offsets = makeOffsetVector(B);
    auto res_offset = makeOffsetVector(res);

    cl::Event evt;
    clblast::GemmBatched<float>(clblast::Layout::kRowMajor, ta, tb, m, n, k, alphas.data(),
                                A.getBuffer()(), a_offset.data(), A_cols, B.data(),
                                b_offsets.data(), B_cols, betas.data(), res.data(),
                                res_offset.data(), C_cols, res.depth, &queue());
    auto c_offset = makeNullOffsetVector(C, res.depth);
    clblast::AxpyBatched<float>(res.rows * res.cols, betas.data(), C.getBuffer()(), c_offset.data(),
                                1, res.data(), res_offset.data(), 1, res.depth, &queue(), &evt());
    if (blocking) evt.wait();

    return res;
  }


  clFTensor clFTensor::batchedGemm(float alpha, bool transpose_a, const clFTensor &A,
                                   bool transpose_b, const clFTensor &B, cl::CommandQueue &queue,
                                   bool blocking) {
    const size_t A_rows = A.getRows(), A_cols = A.getCols(), B_rows = B.getRows(),
                 B_cols = B.getCols();

    if ((transpose_a ? A_rows : A_cols) != (transpose_b ? B_cols : B_rows))
      throw std::invalid_argument("clFTensor::batchedGemm: Matrix size do not match");

    auto ta = transpose_a ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    auto tb = transpose_b ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    size_t m = (transpose_a ? A_cols : A_rows);
    size_t n = (transpose_b ? B_rows : B_cols);
    size_t k = (transpose_a ? A_rows : A_cols);

    clFTensor res((transpose_a ? A_cols : A_rows), (transpose_b ? B_rows : B_cols), B.depth);
    res.fill(0.0f, queue, false);
    cl::Event evt;

    std::vector<float> alphas(A.depth, alpha);
    std::vector<float> betas(res.depth, 0.0f);

    auto a_offset = makeOffsetVector(A);
    auto b_offsets = makeOffsetVector(B);
    auto c_offset = makeOffsetVector(res);

    clblast::GemmBatched<float>(clblast::Layout::kRowMajor, ta, tb, m, n, k, alphas.data(),
                                A.getBuffer()(), a_offset.data(), A_cols, B.data(),
                                b_offsets.data(), B_cols, betas.data(), res.data(), c_offset.data(),
                                res.getCols(), B.depth, &queue(), &evt());
    if (blocking) evt.wait();

    return res;
  }

  clFMatrix clFTensor::sumCollapse(cl::CommandQueue &queue, bool blocking) const {
    clFMatrix result(rows, cols);
    result.fill(0.0f, queue, false);

    if (depth == 0) { return result; }

    for (size_t i = 0; i < depth; i++) {
      clFMatrix mat = (*this)[i];
      if (i == depth - 1) {
        result.ipadd(1.0f, mat, queue, blocking);
      } else {
        result.ipadd(1.0f, mat, queue, false);
      }
    }
    return result;
  }


  clFTensor &clFTensor::iphadamard(const clFTensor &other, cl::CommandQueue &queue, bool blocking) {
    if (rows != other.rows || cols != other.cols || depth != other.depth) {
      throw std::runtime_error("clFTensor::sub: Cannot subtract two tensors with different sizes");
    }

    size_t size = rows * cols;

    cl::Event evt;

    clblast::Had<float>(size * depth, 1.0f, other.data(), other.offset * size, 1, data(),
                        offset * size, 1, 0.0f, data(), offset * size, 1, &queue(), &evt());
    if (blocking) evt.wait();
    return *this;
  }

  void clFTensor::ipscale(float scale, cl::CommandQueue &queue, bool blocking) {
    if (getDepth() == 0 || getRows() == 0 || getCols() == 0) return;

    cl::Event evt;
    clblast::Scal<float>(rows * cols * depth, scale, data(), offset, 1, &queue(), &evt());
    if (blocking) evt.wait();
  }

}   // namespace math
