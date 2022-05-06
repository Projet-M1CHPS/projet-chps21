#include "clFMatrix.hpp"

namespace math {

  std::ostream &operator<<(std::ostream &os, const clFMatrix &m) {
    FloatMatrix mat = m.toFloatMatrix();
    os << mat;
    return os;
  }

  clFMatrix::clFMatrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
    // clblast doesn't support zero-sized operations
    // And this would waste cpu time anyway
    if (size() == 0) return;
    data = cl::Buffer(CL_MEM_READ_WRITE, rows * cols * sizeof(float));
  }

  clFMatrix::clFMatrix(const float *source, size_t rows, size_t cols, cl::CommandQueue &queue,
                       bool blocking)
      : rows(rows), cols(cols) {
    // clblast doesn't support zero-sized operations
    // And this would waste cpu time anyway
    if (size() == 0) return;

    try {
      data = cl::Buffer(CL_MEM_READ_WRITE, rows * cols * sizeof(float));
    } catch (cl::Error &err) {
      std::cerr << "[clFMatrix::clFMatrix] ERROR: " << err.what() << "(" << err.err() << ")"
                << std::endl;
      std::cerr << "Size is: " << rows * cols * sizeof(float) << " bytes." << std::endl;
      std::cerr << "Description here: https://streamhpc.com/blog/2013-04-28/opencl-error-codes/"
                << std::endl;
      throw err;
    }
    queue.enqueueWriteBuffer(data, blocking, 0, rows * cols * sizeof(float), source);
  }

  clFMatrix::clFMatrix(const math::FloatMatrix &matrix, cl::CommandQueue &queue, bool blocking) {
    fromFloatMatrix(matrix, queue, blocking);
  }

  clFMatrix &clFMatrix::operator=(const FloatMatrix &other) {
    // If the matrix is empty, deallocate the buffer if there is one and return immediately
    if (other.getSize() == 0) {
      data = cl::Buffer();
      rows = 0;
      cols = 0;
      return *this;
    }

    // If a buffer is already allocated, and not big enough to store the new matrix, create a new
    // buffer. Else, keep the same buffer
    // Since matrices rarely changes size, this check is worth it
    if (size() != other.getSize()) {
      data = cl::Buffer(CL_MEM_READ_WRITE, other.getRows() * other.getCols() * sizeof(float));
    }
    rows = other.getRows();
    cols = other.getCols();

    enqueueWriteBuffer(data, true, offset * sizeof(float), rows * cols * sizeof(float),
                       (void *) other.getData());
    return *this;
  }

  clFMatrix &clFMatrix::copy(const clFMatrix &other, cl::CommandQueue &queue, bool blocking) {
    if (size() != other.size()) {
      if (offset != 0)
        throw std::runtime_error("clFMatrix::copy: Cannot copy a matrix with a different size, "
                                 "when the destination is a submatrix");
      if (other.size() == 0) data = cl::Buffer();
      else
        data = cl::Buffer(CL_MEM_READ_WRITE, other.rows * other.cols * sizeof(float));
    }

    rows = other.rows;
    cols = other.cols;

    // We need to return if the size is 0 else OpenCL will throw
    if (size() != 0) {
      cl::Event evt;
      queue.enqueueCopyBuffer(other.data, data, other.offset * sizeof(float),
                              offset * sizeof(float), rows * cols * sizeof(float), nullptr, &evt);
      if (blocking) evt.wait();
    }
    return *this;
  }


  clFMatrix::clFMatrix(cl::Buffer &subbuffer, size_t width, size_t height, size_t offset)
      : data(subbuffer), rows(width), cols(height), offset(offset) {}

  clFMatrix::clFMatrix(const cl::Buffer &subbuffer, size_t width, size_t height, size_t offset)
      : data(subbuffer), rows(width), cols(height), offset(offset) {}

  clFMatrix clFMatrix::flatten() const {
    clFMatrix res;
    res.data = data;
    res.rows = rows * cols;
    res.cols = cols == 0 ? 0 : 1;
    res.offset = offset;
    return res;
  }


  /// Conversion functions
  /// Those are especially important since it is easier to manipulate matrices that are on the
  /// host and direct memory access is faster than through OpenCL read and write operations They
  /// may be used for matrix creations (e.g random matrix is created on the host then moved to the
  /// device). As such, they should be considered pri mitives and not utility.

  /// This operation is always blocking to prevent the host memory from being deallocated before
  /// the shift occurs
  void clFMatrix::fromFloatMatrix(const math::FloatMatrix &matrix, cl::CommandQueue &queue,
                                  bool blocking) {
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
      data = cl::Buffer(CL_MEM_READ_WRITE, matrix.getRows() * matrix.getCols() * sizeof(float));
    }

    rows = matrix.getRows();
    cols = matrix.getCols();
    queue.enqueueWriteBuffer(data, blocking, offset * sizeof(float), rows * cols * sizeof(float),
                             (void *) matrix.getData());
  }

  // This operation can be performed non-blocking since the device memory is not deallocated until
  // is it dereferenced
  // This can be useful for shifting multiple matrices from the host
  FloatMatrix clFMatrix::toFloatMatrix(cl::CommandQueue &queue, bool blocking) const {
    FloatMatrix matrix(rows, cols);

    if (size() != 0)
      queue.enqueueReadBuffer(data, blocking, offset * sizeof(float), rows * cols * sizeof(float),
                              (void *) matrix.getData());

    return matrix;
  }

  clFMatrix &clFMatrix::fill(float value, cl::CommandQueue &queue, bool blocking) {
    if (size() == 0) return *this;

    cl::Event evt;
    queue.enqueueFillBuffer(data, value, offset * sizeof(float), rows * cols * sizeof(float),
                            nullptr, &evt);
    if (blocking) evt.wait();
    return *this;
  }

  float clFMatrix::sumReduce(cl::CommandQueue &queue) const {
    if (size() == 0) throw std::runtime_error("Cannot sum an empty matrix");

    // Perform the sum on the platform
    cl::Buffer res_buf(CL_MEM_READ_WRITE, sizeof(float));
    clblast::Asum<float>(size(), res_buf(), 0, data(), offset, 1, &queue());

    // Shift the result to the host
    float res = 0;
    queue.enqueueReadBuffer(res_buf, true, 0, sizeof(float), &res);
    return res;
  }

  float clFMatrix::l2norm(cl::CommandQueue &queue) const {
    // Perform the l2norm on the platform
    cl::Buffer res_buf(CL_MEM_READ_WRITE, sizeof(float));
    clblast::Nrm2<float>(size(), res_buf(), 0, data(), offset, 1, &queue());

    // Shift the result to the host
    float res = 0;
    queue.enqueueReadBuffer(res_buf, true, 0, sizeof(float), &res);
    return res;
  }

  size_t clFMatrix::imax(cl::CommandQueue &queue) const {
    if (size() == 0) throw std::runtime_error("Cannot imax an empty matrix");

    // Perform the imax on the platform
    cl::Buffer res_buf(CL_MEM_READ_WRITE, sizeof(cl_uint));
    clblast::Amax<float>(size(), res_buf(), 0, data(), offset, 1, &queue());

    // Shift the result to the host
    cl_uint res_long = 0;
    queue.enqueueReadBuffer(res_buf, true, 0, sizeof(cl_uint), &res_long);
    return res_long;
  }

  clFMatrix clFMatrix::transpose(cl::CommandQueue &queue, bool blocking) const {
    clFMatrix res(cols, rows);

    // clblast throws if the size is 0 (and we don't want to spend time enqueuing a useless
    // kernel)
    if (size() == 0) return res;

    cl::Event evt;
    clblast::Omatcopy<float>(clblast::Layout::kRowMajor, clblast::Transpose::kYes, rows, cols, 1.0f,
                             data(), offset, cols, res.data(), 0, rows, &queue(), &evt());
    if (blocking) evt.wait();
    return res;
  }

  void clFMatrix::ipadd(float factor, const clFMatrix &other, cl::CommandQueue &queue,
                        bool blocking) {
    if (rows != other.rows or cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }
    cl::Event evt;
    clblast::Axpy<float>(size(), factor, other.data(), other.offset, 1, data(), offset, 1, &queue(),
                         &evt());
    if (blocking) evt.wait();
  }

  clFMatrix clFMatrix::add(float factor, const clFMatrix &other, cl::CommandQueue &queue,
                           bool blocking) const {
    // To avoid copies, we need not use the += operator
    // and directly perform the addition in the result matrix
    // This is the reason behind this code duplicate
    // We could refactor this by creating an external method
    // but this raises issue with the const correctness
    if (rows != other.rows or cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }

    clFMatrix res;
    res.copy(other, queue, false);

    cl::Event evt;
    clblast::Axpy<float>(size(), factor, data(), offset, 1, res.data(), 0, 1, &queue(), &evt());
    if (blocking) evt.wait();
    return res;
  }

  void clFMatrix::ipsub(float factor, const clFMatrix &other, cl::CommandQueue &queue,
                        bool blocking) {
    if (rows != other.rows or cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }

    cl::Event evt;
    clblast::Axpy<float>(size(), -factor, other.data(), other.offset, 1, data(), offset, 1,
                         &queue(), &evt());
    if (blocking) evt.wait();
  }

  clFMatrix clFMatrix::sub(float factor, const clFMatrix &other, cl::CommandQueue &queue,
                           bool blocking) const {
    // To avoid copies, we need not use the += operator
    // and directly perform the addition in the result matrix
    // This is the reason behind this code duplicate
    // We could refactor this by creating an external method
    // but this raises issue with the const correctness
    if (rows != other.rows or cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }

    clFMatrix res;
    res.copy(*this, queue, false);

    cl::Event evt;
    clblast::Axpy<float>(size(), -factor, other.data(), other.offset, 1, res.data(), 0, 1, &queue(),
                         &evt());
    if (blocking) evt.wait();
    return res;
  }

  void clFMatrix::ipscale(float scale, cl::CommandQueue &queue, bool blocking) {
    if (size() == 0) return;

    cl::Event evt;
    clblast::Scal<float>(rows * cols, scale, data(), offset, 1, &queue(), &evt());
    if (blocking) evt.wait();
  }

  clFMatrix clFMatrix::scale(float scale, cl::CommandQueue &queue, bool blocking) const {
    clFMatrix res;
    res.copy(*this, queue, false);

    if (size() == 0) return res;

    cl::Event evt;
    clblast::Scal<float>(rows * cols, scale, res.data(), 0, 1, &queue(), &evt());
    if (blocking) evt.wait();
    return res;
  }

  void clFMatrix::iphadamard(const clFMatrix &other, cl::CommandQueue &queue, bool blocking) const {
    if (rows != other.rows or cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }

    cl::Event evt;
    clblast::Had<float>(rows * cols, 1.0f, data(), offset, 1, other.data(), other.offset, 1, 0.0f,
                        data(), offset, 1, &queue(), &evt());
    if (blocking) evt.wait();
  }

  clFMatrix clFMatrix::hadamard(const clFMatrix &other, cl::CommandQueue &queue,
                                bool blocking) const {
    if (rows != other.rows or cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }

    clFMatrix res;
    res.copy(other, queue, false);
    res.iphadamard(*this, queue, blocking);
    return res;
  }

  clFMatrix clFMatrix::gemm(float alpha, bool transpose_a, const clFMatrix &A, bool transpose_b,
                            const clFMatrix &B, cl::CommandQueue &queue, bool blocking) {
    const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols;

    if ((transpose_a ? A_rows : A_cols) != (transpose_b ? B_cols : B_rows)) {
      throw std::invalid_argument("Matrix size do not match");
    }

    auto ta = transpose_a ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    auto tb = transpose_b ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    size_t m = (transpose_a ? A_cols : A_rows);
    size_t n = (transpose_b ? B_rows : B_cols);
    size_t k = (transpose_a ? A_rows : A_cols);

    clFMatrix res((transpose_a ? A_cols : A_rows), (transpose_b ? B_rows : B_cols));
    cl::Event evt;
    clblast::Gemm<float>(clblast::Layout::kRowMajor, ta, tb, m, n, k, alpha, A.data(), A.offset,
                         A_cols, B.data(), B.offset, B_cols, 0.f, res.data(), 0, res.getCols(),
                         &queue(), &evt());
    if (blocking) evt.wait();

    return res;
  }

  clFMatrix clFMatrix::gemm(float alpha, bool transpose_a, const clFMatrix &A, bool transpose_b,
                            const clFMatrix &B, float beta, const clFMatrix &C,
                            cl::CommandQueue &queue, bool blocking) {
    const size_t A_rows = A.rows, A_cols = A.cols, B_rows = B.rows, B_cols = B.cols,
                 C_rows = C.rows, C_cols = C.cols;

    if (A_cols != B_rows || A_rows != C_rows || B_cols != C_cols) {
      throw std::invalid_argument("Matrix dimensions do not match");
    }

    auto ta = transpose_a ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    auto tb = transpose_b ? clblast::Transpose::kYes : clblast::Transpose::kNo;
    size_t m = (transpose_a ? A_cols : A_rows);
    size_t n = (transpose_b ? B_rows : B_cols);
    size_t k = (transpose_a ? A_rows : A_cols);

    clFMatrix res;
    res.copy(C, queue, false);

    cl::Event evt;
    clblast::Gemm<float>(clblast::Layout::kRowMajor, ta, tb, m, n, k, alpha, A.data(), A.offset,
                         A_cols, B.data(), B.offset, B_cols, beta, res.data(), 0, res.getCols(),
                         &queue(), &evt());
    if (blocking) evt.wait();
    return res;
  }
}   // namespace math