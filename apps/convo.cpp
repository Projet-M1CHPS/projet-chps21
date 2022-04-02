#include "Matrix.hpp"
#include "clUtils/clFMatrix.hpp"
#include <iostream>

using namespace math;
using namespace std;


int main() {
  math::FloatMatrix A(6, 6);
  A.fill(1.f);
  {
    A(0, 0) = 1.f;
    A(0, 1) = 2.f;
    A(0, 2) = 1.f;
    A(0, 3) = 1.f;
    A(0, 4) = 4.f;
    A(0, 5) = 1.f;
    A(1, 0) = 2.f;
    A(1, 1) = 1.f;
    A(1, 2) = 1.f;
    A(1, 3) = 2.f;
    A(1, 4) = 2.f;
    A(1, 5) = 1.f;
    A(2, 0) = 4.f;
    A(2, 1) = 3.f;
    A(2, 2) = 2.f;
    A(2, 3) = 1.f;
    A(2, 4) = 2.f;
    A(2, 5) = 1.f;
    A(3, 0) = 1.f;
    A(3, 1) = 5.f;
    A(3, 2) = 1.f;
    A(3, 3) = 1.f;
    A(3, 4) = 2.f;
    A(3, 5) = 1.f;
    A(4, 0) = 2.f;
    A(4, 1) = 1.f;
    A(4, 2) = 1.f;
    A(4, 3) = 4.f;
    A(4, 4) = 1.f;
    A(4, 5) = 1.f;
    A(5, 0) = 2.f;
    A(5, 1) = 1.f;
    A(5, 2) = 4.f;
    A(5, 3) = 2.f;
    A(5, 4) = 4.f;
    A(5, 5) = 1.f;
  }
  std::cout << "A = \n" << A << std::endl;
  auto a = clFMatrix(A, true);


  math::FloatMatrix B(4, 1);
  {
    B(0, 0) = 2.f;
    B(0, 1) = 1.f;
    B(0, 2) = 0.5f;
    B(0, 3) = 1.5f;
    //B(0, 4) = 0.f;
    //B(0, 5) = 0.f;
    //B(0, 6) = 0.f;
    //B(0, 7) = 0.f;
  }
  std::cout << "B = \n" << B << std::endl;
  auto b = clFMatrix(B, true);

  math::FloatMatrix O(5, 5);
  O.fill(100.f);
  clFMatrix out(O, true);


  cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();


  clblast::Convgemm<float>(clblast::KernelMode::kCrossCorrelation, 1, 6, 6, 2, 2, 0, 0, 1, 1, 1, 1,
                           1, 1, a.getBuffer()(), 0, b.getBuffer()(), 0, out.getBuffer()(), 0,
                           &queue(), nullptr);


  FloatMatrix tmp = out.toFloatMatrix(true);
  cout << "output\n" << tmp << endl;

  return 0;
}