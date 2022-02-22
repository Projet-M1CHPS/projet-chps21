#include "clFMatrix.hpp"

namespace math {

  clFMatrix::clFMatrix(size_t rows, size_t cols, utils::clWrapper &wrapper)
      : rows(rows), cols(cols) {
    // clblast doesn't support zero-sized operations
    // And this would waste cpu time anyway
    if (size() == 0) return;
    data = cl::Buffer(wrapper.getContext(), CL_MEM_READ_WRITE, rows * cols * sizeof(float));
  }

  clFMatrix::clFMatrix(const float *source, size_t rows, size_t cols, utils::clWrapper &wrapper,
                       cl::CommandQueue &queue, bool blocking)
      : rows(rows), cols(cols) {
    // clblast doesn't support zero-sized operations
    // And this would waste cpu time anyway
    if (size() == 0) return;

    data = cl::Buffer(wrapper.getContext(), CL_MEM_READ_WRITE, rows * cols * sizeof(float));
    queue.enqueueWriteBuffer(data, blocking, 0, rows * cols * sizeof(float), source);
  }

  clFMatrix::clFMatrix(const math::FloatMatrix &matrix, utils::clWrapper &wrapper,
                       cl::CommandQueue &queue, bool blocking) {
    fromFloatMatrix(matrix, wrapper, queue, blocking);
  }


  clFMatrix::clFMatrix(const clFMatrix &other, utils::clWrapper &wrapper, cl::CommandQueue &queue,
                       bool blocking) {
    rows = other.rows;
    cols = other.cols;
    // We need to return if the size is 0 else OpenCL will throw
    if (size() == 0) return;

    data = cl::Buffer(wrapper.getContext(), CL_MEM_READ_WRITE, rows * cols * sizeof(float));
    cl::Event evt;
    wrapper.getDefaultQueue().enqueueCopyBuffer(other.data, data, 0, 0, rows * cols * sizeof(float),
                                                nullptr, &evt);
    if (blocking) evt.wait();
  }

  /// Conversion functions
  /// Those are especially important since it is easier to manipulate matrices that are on the host
  /// and direct memory access is faster than through OpenCL read and write operations
  /// They may be used for matrix creations (e.g random matrix is created on the host then moved to
  /// the device). As such, they should be considered primitives and not utility.

  /// This operation is always blocking to prevent the host memory from being deallocated before the
  /// shift occurs
  void clFMatrix::fromFloatMatrix(const math::FloatMatrix &matrix, utils::clWrapper &wrapper,
                                  cl::CommandQueue &queue, bool blocking) {
    // If the matrix is empty, deallocate the buffer if there is one and return immediately
    if (matrix.getSize() == 0) {
      data = cl::Buffer();
      rows = 0;
      cols = 0;
      return;
    }

    // If a buffer is already allocated, and not big enough to store the new matrix, create a new
    // buffer. Else, keep the same buffer
    // Since matrices rarely changes size, this check is worth it
    if (rows * cols != matrix.getRows() * matrix.getCols()) {
      data = cl::Buffer(wrapper.getContext(), CL_MEM_READ_WRITE,
                        matrix.getRows() * matrix.getCols() * sizeof(float));
    }

    rows = matrix.getRows();
    cols = matrix.getCols();
    wrapper.getDefaultQueue().enqueueWriteBuffer(data, blocking, 0, rows * cols * sizeof(float),
                                                 (void *) matrix.getData());
  }

  // This operation can be performed non-blocking since the device memory is not deallocated until
  // is it dereferenced
  // This can be useful for shifting multiple matrices from the host
  FloatMatrix clFMatrix::toFloatMatrix(utils::clWrapper &wrapper, cl::CommandQueue &queue,
                                       bool blocking) const {
    FloatMatrix matrix(rows, cols);

    if (size() != 0)
      queue.enqueueReadBuffer(data, blocking, 0, rows * cols * sizeof(float),
                              (void *) matrix.getData());
    return matrix;
  }

  float clFMatrix::sumReduce(utils::clWrapper &wrapper, cl::CommandQueue &queue) const {
    // Perform the sum on the platform
    cl::Buffer res_buf(wrapper.getContext(), CL_MEM_READ_WRITE, sizeof(float));
    clblast::Asum<float>(size(), res_buf(), 0, data(), 0, 1, &wrapper.getDefaultQueue()());

    // Shift the result to the host
    float res = 0;
    queue.enqueueReadBuffer(res_buf, true, 0, sizeof(float), &res);
    return res;
  }

  float clFMatrix::l2norm(utils::clWrapper &wrapper, cl::CommandQueue &queue) const {
    // Perform the l2norm on the platform
    cl::Buffer res_buf(wrapper.getContext(), CL_MEM_READ_WRITE, sizeof(float));
    clblast::Nrm2<float>(size(), res_buf(), 0, data(), 0, 1, &wrapper.getDefaultQueue()());

    // Shift the result to the host
    float res = 0;
    queue.enqueueReadBuffer(res_buf, true, 0, sizeof(float), &res);
    return res;
  }

  clFMatrix clFMatrix::transpose(utils::clWrapper &wrapper, cl::CommandQueue &queue,
                                 bool blocking) const {
    clFMatrix res(cols, rows, wrapper);

    // clblast throws if the size is 0 (and we don't want to spend time enqueuing a useless kernel)
    if (size() == 0) return res;

    cl::Event evt;
    clblast::Omatcopy<float>(clblast::Layout::kRowMajor, clblast::Transpose::kYes, rows, cols, 1.0f,
                             data(), 0, cols, res.data(), 0, rows, &wrapper.getDefaultQueue()(),
                             &evt());
    if (blocking) evt.wait();
    return res;
  }

  void clFMatrix::ipadd(const clFMatrix &other, utils::clWrapper &wrapper, cl::CommandQueue &queue,
                        bool blocking) {
    if (rows != other.rows or cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }
    cl::Event evt;
    clblast::Axpy<float>(size(), 1.0f, other.data(), 0, 1, data(), 0, 1,
                         &wrapper.getDefaultQueue()(), &evt());
    if (blocking) evt.wait();
  }

  clFMatrix clFMatrix::add(const clFMatrix &other, utils::clWrapper &wrapper,
                           cl::CommandQueue &queue, bool blocking) const {
    // To avoid copies, we need not use the += operator
    // and directly perform the addition in the result matrix
    // This is the reason behind this code duplicate
    // We could refactor this by creating an external method
    // but this raises issue with the const correctness
    if (rows != other.rows or cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }

    clFMatrix res(other, wrapper);
    cl::Event evt;
    clblast::Axpy<float>(size(), 1.0f, data(), 0, 1, res.data(), 0, 1, &wrapper.getDefaultQueue()(),
                         &evt());
    if (blocking) evt.wait();
    return res;
  }

  void clFMatrix::ipsub(const clFMatrix &other, utils::clWrapper &wrapper, cl::CommandQueue &queue,
                        bool blocking) {
    if (rows != other.rows or cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }

    cl::Event evt;
    clblast::Axpy<float>(size(), -1.0f, other.data(), 0, 1, data(), 0, 1,
                         &wrapper.getDefaultQueue()(), &evt());
    if (blocking) evt.wait();
  }

  clFMatrix clFMatrix::sub(const clFMatrix &other, utils::clWrapper &wrapper,
                           cl::CommandQueue &queue, bool blocking) const {
    // To avoid copies, we need not use the += operator
    // and directly perform the addition in the result matrix
    // This is the reason behind this code duplicate
    // We could refactor this by creating an external method
    // but this raises issue with the const correctness
    if (rows != other.rows or cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }

    clFMatrix res(*this, wrapper);
    cl::Event evt;
    clblast::Axpy<float>(size(), -1.0f, other.data(), 0, 1, res.data(), 0, 1,
                         &wrapper.getDefaultQueue()(), &evt());
    if (blocking) evt.wait();
    return res;
  }

  void clFMatrix::ipscale(const float scale, utils::clWrapper &wrapper) {
    clblast::Scal<float>(rows * cols, scale, data(), 0, 1, &wrapper.getDefaultQueue()());
  }

  clFMatrix clFMatrix::scale(const float scale, utils::clWrapper &wrapper) const {
    clFMatrix res(*this, wrapper);

    clblast::Scal<float>(rows * cols, scale, res.data(), 0, 1, &wrapper.getDefaultQueue()());
    return res;
  }

  clFMatrix clFMatrix::iphadamard(const clFMatrix &other, utils::clWrapper &wrapper) const {
    if (rows != other.rows or cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }

    clFMatrix res(rows, cols, wrapper);


    clblast::Had<float>(rows * cols, 1.0f, data(), 0, 1, other.data(), 0, 1, 0.0f, data(), 0, 1,
                        &wrapper.getDefaultQueue()());
    return res;
  }

  clFMatrix clFMatrix::hadamard(const clFMatrix &other, utils::clWrapper &wrapper) const {
    if (rows != other.rows or cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }

    clFMatrix res(rows, cols, wrapper);


    clblast::Had<float>(rows * cols, 1.0f, data(), 0, 1, other.data(), 0, 1, 0.0f, res.data(), 0, 1,
                        &wrapper.getDefaultQueue()());
    return res;
  }

  clFMatrix clFMatrix::matMatProdMatAdd(const clFMatrix &A, const clFMatrix &B, const clFMatrix &C,
                                        utils::clWrapper &wrapper) {
    const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols,
                 C_rows = C.rows, C_cols = C.cols;

    if (A_cols != B_rows || A_rows != C_rows || B_cols != C_cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }

    clFMatrix res(C, wrapper);
    clblast::Gemm<float>(clblast::Layout::kRowMajor, clblast::Transpose::kNo,
                         clblast::Transpose::kNo, A_rows, B_cols, A_cols, 1.f, A.data(), 0, A_cols,
                         B.data(), 0, B_cols, 1.f, res.data(), 0, C_cols,
                         &wrapper.getDefaultQueue()());
    return res;
  }

  clFMatrix clFMatrix::matTransMatProd(const clFMatrix &A, const clFMatrix &B,
                                       utils::clWrapper &wrapper) {
    const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols;

    if (A_rows != B_rows) { throw std::invalid_argument("Matrix dimensions do not match"); }

    clFMatrix res(A_cols, B_cols, wrapper);
    clblast::Gemm<float>(clblast::Layout::kRowMajor, clblast::Transpose::kYes,
                         clblast::Transpose::kNo, A_rows, B_cols, A_cols, 1.f, A.data(), 0, A_cols,
                         B.data(), 0, B_cols, 0.f, res.data(), 0, B_cols,
                         &wrapper.getDefaultQueue()());

    return res;
  }

  clFMatrix clFMatrix::matMatTransProd(const clFMatrix &A, const clFMatrix &B,
                                       utils::clWrapper &wrapper) {
    const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols;

    if (A_cols != B_cols) { throw std::invalid_argument("Matrix dimensions do not match"); }

    clFMatrix res(A_cols, B_cols, wrapper);
    clblast::Gemm<float>(clblast::Layout::kRowMajor, clblast::Transpose::kNo,
                         clblast::Transpose::kYes, A_rows, B_cols, A_cols, 1.f, A.data(), 0, A_cols,
                         B.data(), 0, B_cols, 0.f, res.data(), 0, B_cols,
                         &wrapper.getDefaultQueue()());
    return res;
  }

  clFMatrix clFMatrix::mul(const clFMatrix &other, utils::clWrapper &wrapper) const {
    const size_t other_rows = other.rows, other_cols = other.cols;
    if (cols != other_rows) { throw std::invalid_argument("Matrix dimensions do not match"); }

    clFMatrix res(rows, other_cols, wrapper);

    clblast::Gemm<float>(clblast::Layout::kRowMajor, clblast::Transpose::kNo,
                         clblast::Transpose::kNo, rows, other_cols, cols, 1.f, data(), 0, cols,
                         other.data(), 0, other_cols, 0.f, res.data(), 0, other_cols,
                         &wrapper.getDefaultQueue()());
    return res;
  }

  clFMatrix clFMatrix::mul(const bool transpose_a, const clFMatrix &A, const bool transpose_b,
                           const clFMatrix &B, utils::clWrapper &wrapper, const float alpha) {
    const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols;

    if ((transpose_a ? A_rows : A_cols) != (transpose_b ? B_cols : B_rows)) {
      throw std::invalid_argument("Matrix size do not match");
    }

    auto ta = transpose_a ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    auto tb = transpose_b ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    size_t m = (transpose_a ? A_cols : A_rows);
    size_t n = (transpose_b ? B_rows : B_cols);
    size_t k = (transpose_a ? A_rows : A_cols);

    clFMatrix res((transpose_a ? A_cols : A_rows), (transpose_b ? B_rows : B_cols), wrapper);
    clblast::Gemm<float>(clblast::Layout::kRowMajor, ta, tb, m, n, k, alpha, A.data(), 0, A_cols,
                         B.data(), 0, B_cols, 0.f, res.data(), 0, res.getCols(),
                         &wrapper.getDefaultQueue()());

    return res;
  }
}   // namespace math