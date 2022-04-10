#include "CNN.hpp"
#include "CNNLayer.hpp"
#include "CNNTopology.hpp"
#include "Matrix.hpp"
#include "clUtils/clFMatrix.hpp"
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


void testConvolutionalLayer1Branch() {
  // 1 branch 2 filter/branch 2 input/branch
  const size_t number_filter = 2;
  const size_t number_input = 2;

  nnet::CNNConvolutionLayer layer({5, 5}, {2, 2}, number_filter, af::ActivationFunctionType::relu,
                                  1, 1);

  FloatMatrix f1(2, 2);
  {
    f1(0, 0) = 2.f;
    f1(0, 1) = 1.f;
    f1(1, 0) = 0.5f;
    f1(1, 1) = 1.5f;
  }
  FloatMatrix f2(2, 2);
  {
    f2(0, 0) = 1.f;
    f2(0, 1) = 1.f;
    f2(1, 0) = 1.f;
    f2(1, 1) = 1.f;
  }
  auto &filter = layer.getFilter();
  filter.getMatrix(0) = f1;
  filter.getMatrix(1) = f2;

  for (size_t i = 0; i < filter.getZ(); i++)
    std::cout << "filter " << i << " : \n" << filter.getMatrix(i).toFloatMatrix(true) << std::endl;

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

  clFTensor input_tensor(6, 6, number_input);
  for (size_t i = 0; i < input_tensor.getZ(); i++) input_tensor.getMatrix(i) = input;
  for (size_t i = 0; i < input_tensor.getZ(); i++)
    std::cout << "input " << i << " : \n"
              << input_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.compute(input_tensor);

  for (size_t i = 0; i < output_tensor.getZ(); i++)
    std::cout << "output " << i << " : \n"
              << output_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;

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

void testConvolutionalLayerXBranch() {
  // 3 branch 2 filter/branch 3input/branch
  nnet::CNNConvolutionLayer layer({5, 5}, {2, 2}, 2, af::ActivationFunctionType::relu, 1, 3);

  FloatMatrix f1(2, 2);
  {
    f1(0, 0) = 2.f;
    f1(0, 1) = 1.f;
    f1(1, 0) = 0.5f;
    f1(1, 1) = 1.5f;
  }
  FloatMatrix f2(2, 2);
  {
    f2(0, 0) = 1.f;
    f2(0, 1) = 1.f;
    f2(1, 0) = 1.f;
    f2(1, 1) = 1.f;
  }
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

  auto &filter = layer.getFilter();
  filter.getMatrix(0) = f1;
  filter.getMatrix(1) = f2;
  filter.getMatrix(2) = f1;
  filter.getMatrix(3) = f2;
  filter.getMatrix(4) = f1;
  filter.getMatrix(5) = f2;
  for (size_t i = 0; i < filter.getZ(); i++)
    std::cout << "filter " << i << " : \n" << filter.getMatrix(i).toFloatMatrix(true) << std::endl;

  clFTensor input_tensor(6, 6, 6);
  for (size_t i = 0; i < input_tensor.getZ(); i++) input_tensor.getMatrix(i) = input;
  for (size_t i = 0; i < input_tensor.getZ(); i++)
    std::cout << "input " << i << " : \n"
              << input_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.compute(input_tensor);

  for (size_t i = 0; i < output_tensor.getZ(); i++)
    std::cout << "output " << i << " : \n"
              << output_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;

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

  clFTensor input_tensor(6, 6, 4);
  for (size_t i = 0; i < input_tensor.getZ(); i++) input_tensor.getMatrix(i) = input;
  for (size_t i = 0; i < input_tensor.getZ(); i++)
    std::cout << "input " << i << " : \n"
              << input_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.compute(input_tensor);

  for (size_t i = 0; i < output_tensor.getZ(); i++)
    std::cout << "output " << i << " : \n"
              << output_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;

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
}

void testMaxPoolingLayerBP() {
  nnet::CNNMaxPoolingLayer layer({4, 4}, {3, 3}, 1);
  nnet::CNNStorageBPMaxPooling storage({6, 6});

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

  clFTensor input_tensor(6, 6, 4);
  for (size_t i = 0; i < input_tensor.getZ(); i++) input_tensor.getMatrix(i) = input;
  for (size_t i = 0; i < input_tensor.getZ(); i++)
    std::cout << "input " << i << " : \n"
              << input_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.computeForward(input_tensor, storage);

  for (size_t i = 0; i < output_tensor.getZ(); i++)
    std::cout << "output " << i << " : \n"
              << output_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;


  clFTensor errors_tensor(4, 4, 4);
  for (size_t i = 0; i < errors_tensor.getZ(); i++) {
    errors_tensor.getMatrix(i).fill(1.f, utils::cl_wrapper.getDefaultQueue(), true);
  }

  clFTensor errors_input = layer.computeBackward(errors_tensor, storage);

  for (size_t i = 0; i < errors_input.getZ(); i++)
    std::cout << "output " << i << " : \n"
              << errors_input.getMatrix(i).toFloatMatrix(true) << std::endl;

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

  clFTensor input_tensor(6, 6, 4);
  for (size_t i = 0; i < input_tensor.getZ(); i++) input_tensor.getMatrix(i) = input;
  for (size_t i = 0; i < input_tensor.getZ(); i++)
    std::cout << "input " << i << " : \n"
              << input_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.compute(input_tensor);

  for (size_t i = 0; i < output_tensor.getZ(); i++)
    std::cout << "output " << i << " : \n"
              << output_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;

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
}

void testAvgPoolingLayerBP() {
  nnet::CNNAvgPoolingLayer layer({4, 4}, {3, 3}, 1);
  nnet::CNNStorageBPAvgPooling storage({6, 6});

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

  clFTensor input_tensor(6, 6, 4);
  for (size_t i = 0; i < input_tensor.getZ(); i++) input_tensor.getMatrix(i) = input;
  for (size_t i = 0; i < input_tensor.getZ(); i++)
    std::cout << "input " << i << " : \n"
              << input_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.computeForward(input_tensor, storage);

  for (size_t i = 0; i < output_tensor.getZ(); i++)
    std::cout << "output " << i << " : \n"
              << output_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;

  clFTensor errors_tensor(4, 4, 4);
  for (size_t i = 0; i < errors_tensor.getZ(); i++) {
    errors_tensor.getMatrix(i).fill(1.f, utils::cl_wrapper.getDefaultQueue(), true);
  }

  clFTensor errors_input = layer.computeBackward(errors_tensor, storage);

  for (size_t i = 0; i < errors_input.getZ(); i++)
    std::cout << "output " << i << " : \n"
              << errors_input.getMatrix(i).toFloatMatrix(true) << std::endl;

  /* error input
   * 0.11 0.22 0.33 0.33 0.22 0.11
   * 0.22 0.44 0.66 0.66 0.44 0.22
   * 0.33 0.66 1    1    0.66 0.33
   * 0.33 0.66 1    1    0.66 0.33
   * 0.22 0.44 0.66 0.66 0.44 0.22
   * 0.11 0.22 0.33 0.33 0.22 0.11
   * */
}

void testPrediction1Branch() {
  std::string str_topology("6 6 relu convolution 1 2 2 1 0 pooling max 2 2 1");
  // std::string str_topology("6 6 relu convolution 1 2 2 1 0");
  auto topology = nnet::stringToTopology(str_topology);
  std::cout << topology << std::endl;

  nnet::CNN cnn;
  cnn.setTopology(topology);

  FloatMatrix f(2, 2);
  {
    f(0, 0) = 2.f;
    f(0, 1) = 1.f;
    f(1, 0) = 0.5f;
    f(1, 1) = 1.5f;
  }
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

  const auto &layer = cnn.layers[0].get();
  const auto &layer_convolution = dynamic_cast<const nnet::CNNConvolutionLayer *>(layer);
  auto &filter = layer_convolution->getFilter();
  for (size_t i = 0; i < filter.getZ(); i++) filter.getMatrix(i) = f;

  clFTensor input_tensor(6, 6, 2);
  for (size_t i = 0; i < input_tensor.getZ(); i++) input_tensor.getMatrix(i) = input;
  for (size_t i = 0; i < input_tensor.getZ(); i++)
    std::cout << "input " << i << " : \n"
              << input_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;


  clFTensor output_tensor = cnn.predict(input_tensor);

  for (size_t i = 0; i < output_tensor.getZ(); i++)
    std::cout << "output " << i << " : \n"
              << output_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;

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

void testPredictionXBranch() {
  // std::string str_topology("6 6 relu convolution 2 2 2 1 0");
  // std::string str_topology("6 6 relu convolution 2 2 2 1 0 pooling max 2 2 1");
  // std::string str_topology("6 6 relu convolution 2 2 2 1 0 convolution 2 2 2 1 0");
  std::string str_topology(
          "6 6 relu convolution 2 2 2 1 0 convolution 2 2 2 1 0 pooling max 2 2 1");
  auto topology = nnet::stringToTopology(str_topology);
  std::cout << topology << std::endl;

  nnet::CNN cnn;
  cnn.setTopology(topology);

  FloatMatrix f1(2, 2);
  {
    f1(0, 0) = 2.f;
    f1(0, 1) = 1.f;
    f1(1, 0) = 0.5f;
    f1(1, 1) = 1.5f;
  }
  FloatMatrix f2(2, 2);
  {
    f2(0, 0) = 2.f;
    f2(0, 1) = 1.f;
    f2(1, 0) = 0.f;
    f2(1, 1) = 1.f;
  }
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

  const auto &layer1 = dynamic_cast<const nnet::CNNConvolutionLayer *>(cnn.layers[0].get());
  auto &filter1 = layer1->getFilter();
  for (size_t i = 0; i < filter1.getZ(); i++) filter1.getMatrix(i) = f1;

  const auto &layer2 = dynamic_cast<const nnet::CNNConvolutionLayer *>(cnn.layers[1].get());
  auto &filter2 = layer2->getFilter();
  for (size_t i = 0; i < filter2.getZ(); i++) filter2.getMatrix(i) = f2;

  clFTensor input_tensor(6, 6, 2);
  for (size_t i = 0; i < input_tensor.getZ(); i++) input_tensor.getMatrix(i) = input;
  for (size_t i = 0; i < input_tensor.getZ(); i++)
    std::cout << "input " << i << " : \n"
              << input_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;


  clFTensor output_tensor = cnn.predict(input_tensor);

  for (size_t i = 0; i < output_tensor.getZ(); i++)
    std::cout << "output " << i << " : \n"
              << output_tensor.getMatrix(i).toFloatMatrix(true) << std::endl;

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

  // testConvolutionalLayer1Branch();
  // testConvolutionalLayerXBranch();

  // testMaxPoolingLayer();
  testMaxPoolingLayerBP();

  // testAvgPoolingLayer();
  // testAvgPoolingLayerBP();

  // testPrediction1Branch();
  // testPredictionXBranch();

  return 0;
}