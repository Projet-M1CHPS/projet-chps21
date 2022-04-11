#include "CNN.hpp"
#include "CNNLayer.hpp"
#include "CNNTopology.hpp"
#include "math/Matrix.hpp"
#include "math/clFMatrix.hpp"
#include <iostream>

using namespace math;


void testConvo() {
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


  math::FloatMatrix B(1, 8);
  {
    B(0, 0) = 2.f;
    B(0, 1) = 1.f;
    B(0, 2) = 0.5f;
    B(0, 3) = 1.5f;
    B(0, 4) = 0.f;
    B(0, 5) = 0.f;
    B(0, 6) = 0.f;
    B(0, 7) = 0.f;
  }
  std::cout << "B = \n" << B << std::endl;
  auto b = clFMatrix(B, true);

  math::FloatMatrix O(50, 4);
  O.fill(100.f);
  clFMatrix out(O, true);


  cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();


  clblast::Convgemm<float>(clblast::KernelMode::kCrossCorrelation, 1, 3, 6, 2, 2, 0, 0, 1, 1, 1, 1,
                           2, 2, a.getBuffer()(), 0, b.getBuffer()(), 0, out.getBuffer()(), 0,
                           &queue(), nullptr);

  queue.finish();

  FloatMatrix tmp = out.toFloatMatrix(true);
  std::cout << "output\n" << tmp << std::endl;
}


void testConvolutionalLayer() {
  nnet::CNNConvolutionLayer layer({5, 5}, {2, 2}, af::ActivationFunctionType::relu, 1, 0);

  auto &filter = layer.getFilter();
  std::cout << "filter : \n" << filter.getMatrix() << std::endl;

  math::FloatMatrix input(6, 6);
  {
    input(0, 0) = 1.f;
    input(0, 1) = 2.f;
    input(0, 2) = 1.f;
    input(0, 3) = 1.f;
    input(0, 4) = 4.f;
    input(0, 5) = 1.f;
    input(1, 0) = 2.f;
    input(1, 1) = 1.f;
    input(1, 2) = 1.f;
    input(1, 3) = 2.f;
    input(1, 4) = 2.f;
    input(1, 5) = 1.f;
    input(2, 0) = 4.f;
    input(2, 1) = 3.f;
    input(2, 2) = 2.f;
    input(2, 3) = 1.f;
    input(2, 4) = 2.f;
    input(2, 5) = 1.f;
    input(3, 0) = 1.f;
    input(3, 1) = 5.f;
    input(3, 2) = 1.f;
    input(3, 3) = 1.f;
    input(3, 4) = 2.f;
    input(3, 5) = 1.f;
    input(4, 0) = 2.f;
    input(4, 1) = 1.f;
    input(4, 2) = 1.f;
    input(4, 3) = 4.f;
    input(4, 4) = 1.f;
    input(4, 5) = 1.f;
    input(5, 0) = 2.f;
    input(5, 1) = 1.f;
    input(5, 2) = 4.f;
    input(5, 3) = 2.f;
    input(5, 4) = 4.f;
    input(5, 5) = 1.f;
  }
  std::cout << "input : \n" << input << std::endl;

  clFMatrix _input(input, true);

  clFMatrix output = layer.compute(_input);

  math::FloatMatrix tmp_output = output.toFloatMatrix(true);
  std::cout << "output : \n" << tmp_output << std::endl;

  /* input
   * 1 2 1 1 4 1
   * 2 1 1 2 2 1
   * 4 3 2 1 2 1
   * 1 5 1 1 2 1
   * 2 1 1 4 1 1
   * 2 1 4 2 4 1
   *
   * filter
   * 2   1
   * 0.5 1.5
   *
   * error input
   * 6.5  7   6.5 10  11.5
   * 11.5 7.5 6.5 9.5 7.5
   * 19   12  7   7.5 7.5
   * 9.5  13  9.5 7.5 7
   * 7.5 9.5 11 16 6.5
   * */

  /*nnet::CNNStorageBPConvolution storage({6, 6}, {5, 5}, {2, 2}, 1);

  layer.computeForward(input, storage);
  std::cout << "compute forward : \n" << storage.output << std::endl;
  for (auto &i : storage.output) { i = 1.f; }
  layer.computeBackward(input, storage);
  std::cout << "error input : \n" << storage.errorInput << std::endl;
  std::cout << "error filter : \n" << storage.errorFilter << std::endl;*/

  /* error input
   *
   * 2 3 3 3 3 1
   * 2.5 5 5 5 5 2.5
   * 2.5 5 5 5 5 2.5
   * 2.5 5 5 5 5 2.5
   * 2.5 5 5 5 5 2.5
   * 0.5 2 2 2 2 1.5
   *
   * error filter
   * 48 43
   * 52 46
   * */
}

void testMaxPoolingLayer() {
  nnet::CNNMaxPoolingLayer layer({4, 4}, {3, 3}, 1);

  math::FloatMatrix input(6, 6);
  {
    input(0, 0) = 1.f;
    input(0, 1) = 2.f;
    input(0, 2) = 1.f;
    input(0, 3) = 1.f;
    input(0, 4) = 4.f;
    input(0, 5) = 1.f;
    input(1, 0) = 2.f;
    input(1, 1) = 1.f;
    input(1, 2) = 1.f;
    input(1, 3) = 2.f;
    input(1, 4) = 2.f;
    input(1, 5) = 1.f;
    input(2, 0) = 4.f;
    input(2, 1) = 3.f;
    input(2, 2) = 2.f;
    input(2, 3) = 1.f;
    input(2, 4) = 2.f;
    input(2, 5) = 1.f;
    input(3, 0) = 1.f;
    input(3, 1) = 5.f;
    input(3, 2) = 1.f;
    input(3, 3) = 1.f;
    input(3, 4) = 2.f;
    input(3, 5) = 1.f;
    input(4, 0) = 2.f;
    input(4, 1) = 1.f;
    input(4, 2) = 1.f;
    input(4, 3) = 4.f;
    input(4, 4) = 1.f;
    input(4, 5) = 1.f;
    input(5, 0) = 2.f;
    input(5, 1) = 1.f;
    input(5, 2) = 4.f;
    input(5, 3) = 2.f;
    input(5, 4) = 4.f;
    input(5, 5) = 1.f;
  }
  std::cout << input << std::endl;

  clFMatrix _input(input, true);

  clFMatrix output = layer.compute(_input);

  math::FloatMatrix tmp_output = output.toFloatMatrix(true);
  std::cout << "output : \n" << tmp_output << std::endl;

  /* input
   * 1 2 1 1 4 1
   * 2 1 1 2 2 1
   * 4 3 2 1 2 1
   * 1 5 1 1 2 1
   * 2 1 1 4 1 1
   * 2 1 4 2 4 1
   *
   * output
   * 4 3 4 4
   * 5 5 2 2
   * 5 5 4 4
   * 5 5 4 4
   * */

  /*nnet::CNNStorageBPMaxPooling storage({6, 6}, {4, 4});

  layer.computeForward(input, storage);
  std::cout << "compute forward : \n" << storage.output << std::endl;
  for (auto &i : storage.output) { i = 1.f; }
  for (auto &i : storage.errorInput) { i = 0.f; }
  layer.computeBackward(input, storage);
  std::cout << "error input : \n" << storage.errorInput << std::endl;*/

  /* error input
   * 0 0 0 0 2 0
   * 0 0 0 2 0 0
   * 1 1 0 0 0 0
   * 0 6 0 0 0 0
   * 0 0 0 4 0 0
   * 0 0 0 0 0 0
   * */
}

void testAvgPoolingLayer() {
  nnet::CNNAvgPoolingLayer layer({4, 4}, {3, 3}, 1);

  math::FloatMatrix input(6, 6);
  {
    input(0, 0) = 1.f;
    input(0, 1) = 2.f;
    input(0, 2) = 1.f;
    input(0, 3) = 1.f;
    input(0, 4) = 4.f;
    input(0, 5) = 1.f;
    input(1, 0) = 2.f;
    input(1, 1) = 1.f;
    input(1, 2) = 1.f;
    input(1, 3) = 2.f;
    input(1, 4) = 2.f;
    input(1, 5) = 1.f;
    input(2, 0) = 4.f;
    input(2, 1) = 3.f;
    input(2, 2) = 2.f;
    input(2, 3) = 1.f;
    input(2, 4) = 2.f;
    input(2, 5) = 1.f;
    input(3, 0) = 1.f;
    input(3, 1) = 5.f;
    input(3, 2) = 1.f;
    input(3, 3) = 1.f;
    input(3, 4) = 2.f;
    input(3, 5) = 1.f;
    input(4, 0) = 2.f;
    input(4, 1) = 1.f;
    input(4, 2) = 1.f;
    input(4, 3) = 4.f;
    input(4, 4) = 1.f;
    input(4, 5) = 1.f;
    input(5, 0) = 2.f;
    input(5, 1) = 1.f;
    input(5, 2) = 4.f;
    input(5, 3) = 2.f;
    input(5, 4) = 4.f;
    input(5, 5) = 1.f;
  }
  std::cout << "input :\n" << input << std::endl;

  clFMatrix _input(input, true);

  clFMatrix output = layer.compute(_input);

  math::FloatMatrix tmp_output = output.toFloatMatrix(true);
  std::cout << "output : \n" << tmp_output << std::endl;

  /* input
   * 1 2 1 1 4 1
   * 2 1 1 2 2 1
   * 4 3 2 1 2 1
   * 1 5 1 1 2 1
   * 2 1 1 4 1 1
   * 2 1 4 2 4 1
   *
   * output
   * 1.8 1.5 1.7 1.6
   * 2.2 1.8 1.5 1.4
   * 2.2 2.1 1.6 1.5
   * 2   2.2 2.2 1.8
   * */

  /*nnet::CNNStorageBPAvgPooling storage({6, 6}, {4, 4});

  layer.computeForward(input, storage);
  std::cout << "compute forward : \n" << storage.output << std::endl;
  for (auto &i : storage.output) { i = 1.f; }
  layer.computeBackward(input, storage);
  std::cout << "error input : \n" << storage.errorInput << std::endl;*/

  /* error input
   * 0.11 0.22 0.33 0.33 0.22 0.11
   * 0.22 0.44 0.66 0.66 0.44 0.22
   * 0.33 0.66 1    1    0.66 0.33
   * 0.33 0.66 1    1    0.66 0.33
   * 0.22 0.44 0.66 0.66 0.44 0.22
   * 0.11 0.22 0.33 0.33 0.22 0.11
   * */
}

void testPredictionOneBranch() {
  std::string str_topo("6 6 relu convolution 1 2 2 1 0 pooling max 2 2 1");
  auto topo = nnet::stringToTopology(str_topo);
  std::cout << topo << std::endl;

  nnet::CNN cnn;
  cnn.setTopology(topo);

  math::FloatMatrix input(6, 6);
  {
    input(0, 0) = 1.f;
    input(0, 1) = 2.f;
    input(0, 2) = 1.f;
    input(0, 3) = 1.f;
    input(0, 4) = 4.f;
    input(0, 5) = 1.f;
    input(1, 0) = 2.f;
    input(1, 1) = 1.f;
    input(1, 2) = 1.f;
    input(1, 3) = 2.f;
    input(1, 4) = 2.f;
    input(1, 5) = 1.f;
    input(2, 0) = 4.f;
    input(2, 1) = 3.f;
    input(2, 2) = 2.f;
    input(2, 3) = 1.f;
    input(2, 4) = 2.f;
    input(2, 5) = 1.f;
    input(3, 0) = 1.f;
    input(3, 1) = 5.f;
    input(3, 2) = 1.f;
    input(3, 3) = 1.f;
    input(3, 4) = 2.f;
    input(3, 5) = 1.f;
    input(4, 0) = 2.f;
    input(4, 1) = 1.f;
    input(4, 2) = 1.f;
    input(4, 3) = 4.f;
    input(4, 4) = 1.f;
    input(4, 5) = 1.f;
    input(5, 0) = 2.f;
    input(5, 1) = 1.f;
    input(5, 2) = 4.f;
    input(5, 3) = 2.f;
    input(5, 4) = 4.f;
    input(5, 5) = 1.f;
  }
  math::clFMatrix output(16, 1);

  std::cout << input << std::endl;

  math::clFMatrix tmp_input(input);
  cnn.predict(tmp_input, output);

  math::FloatMatrix tmp_output = output.toFloatMatrix(true);
  std::cout << "output :\n" << tmp_output << std::endl;

  /* input
   * 1 2 1 1 4 1
   * 2 1 1 2 2 1
   * 4 3 2 1 2 1
   * 1 5 1 1 2 1
   * 2 1 1 4 1 1
   * 2 1 4 2 4 1
   *
   * filter
   * 2   1
   * 0.5 1.5
   *
   * output
   * 11.5
   * 7.5
   * 10
   * 11.5
   * 19
   * 12
   * 9.5
   * 9.5
   * 19
   * 13
   * 9.5
   * 7.5
   * 13
   * 13
   * 16
   * 16
   * */
}

int main() {
  utils::clWrapper::initOpenCL(*utils::clWrapper::makeDefault());
  // testConvo();

  // testConvolutionalLayer();
  // testMaxPoolingLayer();
  // testAvgPoolingLayer();

  testPredictionOneBranch();

  return 0;
}