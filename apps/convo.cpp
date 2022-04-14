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
    A(0, 1) = 1.f;
    A(0, 2) = 1.f;
    A(0, 3) = 0.f;
    A(0, 4) = 1.f;
    A(0, 5) = 2.f;

    A(1, 0) = 0.f;
    A(1, 1) = 0.f;
    A(1, 2) = 2.f;
    A(1, 3) = 1.f;
    A(1, 4) = 1.f;
    A(1, 5) = 2.f;

    A(2, 0) = 3.f;
    A(2, 1) = 1.f;
    A(2, 2) = 0.f;
    A(2, 3) = 1.f;
    A(2, 4) = 1.f;
    A(2, 5) = 2.f;

    A(3, 0) = 2.f;
    A(3, 1) = 0.f;
    A(3, 2) = 2.f;
    A(3, 3) = 1.f;
    A(3, 4) = 1.f;
    A(3, 5) = 2.f;

    A(4, 0) = 1.f;
    A(4, 1) = 1.f;
    A(4, 2) = 1.f;
    A(4, 3) = 1.f;
    A(4, 4) = 1.f;
    A(4, 5) = 2.f;

    A(5, 0) = 2.f;
    A(5, 1) = 2.f;
    A(5, 2) = 2.f;
    A(5, 3) = 2.f;
    A(5, 4) = 2.f;
    A(5, 5) = 2.f;
  }
  std::cout << "A = \n" << A << std::endl;
  auto img = clFMatrix(A, true);


  math::FloatMatrix B(3, 3);
  {
    B(0, 0) = 5.f;
    B(0, 1) = 2.f;
    B(0, 2) = 2.f;

    B(1, 0) = 2.f;
    B(1, 1) = 3.f;
    B(1, 2) = 1.f;

    B(2, 0) = 1.f;
    B(2, 1) = 2.f;
    B(2, 2) = 2.f;
  }
  std::cout << "B = \n" << B << std::endl;
  auto filter = clFMatrix(B, true);

  math::FloatMatrix O(2, 2);
  O.fill(100.f);
  clFMatrix out(O, true);


  cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();


  clblast::Convgemm<float>(clblast::KernelMode::kCrossCorrelation, 1, 6, 6, 3, 3, 0, 0, 3, 3, 1, 1,
                           1, 1, img.getBuffer()(), 0, filter.getBuffer()(), 0, out.getBuffer()(),
                           0, &queue(), nullptr);

  clblast::Convgemm<float>(clblast::KernelMode::kCrossCorrelation, 1, 3, 3, 2, 2, 2, 2, 1, 1, 2, 2,
                           1, 1, filter.getBuffer()(), 0, out.getBuffer()(), 0, img.getBuffer()(),
                           0, &queue(), nullptr);

  queue.finish();


  FloatMatrix tmp = out.toFloatMatrix(true);
  std::cout << "output\n" << tmp << std::endl;

  std::cout << "aaaa = \n" << img.toFloatMatrix(true) << std::endl;
}


void testConvolutionalLayer1Branch() {
  // 1 branch 2 filter/branch 2 input/branch
  const size_t number_filter = 2;
  const size_t number_input = 3;

  nnet::CNNConvolutionLayer layer({5, 5}, {2, 2}, number_filter, af::ActivationFunctionType::relu,
                                  1);

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
  filter[0] = f1;
  filter[1] = f2;

  for (size_t i = 0; i < filter.getDepth(); i++)
    std::cout << "filter " << i << " : \n" << filter[i].toFloatMatrix(true) << std::endl;

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
  for (size_t i = 0; i < input_tensor.getDepth(); i++) input_tensor[i] = input;
  for (size_t i = 0; i < input_tensor.getDepth(); i++)
    std::cout << "input " << i << " : \n" << input_tensor[i].toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.compute(input_tensor);

  for (size_t i = 0; i < output_tensor.getDepth(); i++)
    std::cout << "output " << i << " : \n" << output_tensor[i].toFloatMatrix(true) << std::endl;

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

void testConvolutionalLayer1BranchBP() {
  // 1 branch 2 filter/branch 2 input/branch
  const size_t number_filter = 1;
  const size_t number_input = 1;

  nnet::CNNConvolutionLayer layer({5, 5}, {2, 2}, number_filter, af::ActivationFunctionType::relu,
                                  1);

  nnet::CNNStorageBPConvolution storage;

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
  filter[0] = f1;
  // filter[1] = f2;

  for (size_t i = 0; i < filter.getDepth(); i++)
    std::cout << "filter " << i << " : \n" << filter[i].toFloatMatrix(true) << std::endl;

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
  for (size_t i = 0; i < input_tensor.getDepth(); i++) input_tensor[i] = input;
  for (size_t i = 0; i < input_tensor.getDepth(); i++)
    std::cout << "input " << i << " : \n" << input_tensor[i].toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.computeForward(input_tensor, storage);

  for (size_t i = 0; i < output_tensor.getDepth(); i++)
    std::cout << "output " << i << " : \n" << output_tensor[i].toFloatMatrix(true) << std::endl;

  storage.input = clFTensor(6, 6, 1);
  storage.input[0] = input;

  storage.errorFilter = clFTensor(2, 2, 1);

  for (size_t i = 0; i < storage.input.getDepth(); i++)
    std::cout << "input storage " << i << " : \n"
              << storage.input[i].toFloatMatrix(true) << std::endl;

  clFTensor errors_tensor(5, 5, number_input);
  for (size_t i = 0; i < errors_tensor.getDepth(); i++) {
    errors_tensor[i].fill(1.f, utils::cl_wrapper.getDefaultQueue(), true);
  }

  clFTensor errors_input = layer.computeBackward(errors_tensor, storage);

  for (size_t i = 0; i < storage.errorFilter.getDepth(); i++)
    std::cout << "error_filter " << i << " : \n"
              << storage.errorFilter[i].toFloatMatrix(true) << std::endl;

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
  nnet::CNNConvolutionLayer layer({5, 5}, {2, 2}, 2, af::ActivationFunctionType::relu, 2);

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
  FloatMatrix f3(2, 2);
  {
    f3(0, 0) = 2.f;
    f3(0, 1) = 2.f;
    f3(1, 0) = 2.f;
    f3(1, 1) = 2.f;
  }
  FloatMatrix f4(2, 2);
  {
    f4(0, 0) = 4.f;
    f4(0, 1) = 4.f;
    f4(1, 0) = 4.f;
    f4(1, 1) = 4.f;
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
  math::FloatMatrix input2(6, 6);
  {
    input2(0, 0) = 100.f;
    input2(0, 1) = 2.f;
    input2(0, 2) = 1.f;
    input2(0, 3) = 1.f;
    input2(0, 4) = 4.f;
    input2(0, 5) = 1.f;
    input2(1, 0) = 2.f;
    input2(1, 1) = 1.f;
    input2(1, 2) = 1.f;
    input2(1, 3) = 2.f;
    input2(1, 4) = 2.f;
    input2(1, 5) = 1.f;
    input2(2, 0) = 4.f;
    input2(2, 1) = 3.f;
    input2(2, 2) = 2.f;
    input2(2, 3) = 1.f;
    input2(2, 4) = 2.f;
    input2(2, 5) = 1.f;
    input2(3, 0) = 1.f;
    input2(3, 1) = 5.f;
    input2(3, 2) = 1.f;
    input2(3, 3) = 1.f;
    input2(3, 4) = 2.f;
    input2(3, 5) = 1.f;
    input2(4, 0) = 2.f;
    input2(4, 1) = 1.f;
    input2(4, 2) = 1.f;
    input2(4, 3) = 4.f;
    input2(4, 4) = 1.f;
    input2(4, 5) = 1.f;
    input2(5, 0) = 2.f;
    input2(5, 1) = 1.f;
    input2(5, 2) = 4.f;
    input2(5, 3) = 2.f;
    input2(5, 4) = 4.f;
    input2(5, 5) = 1.f;
  }

  auto &filter = layer.getFilter();
  filter[0] = f1;
  filter[1] = f2;
  filter[2] = f3;
  filter[3] = f4;

  std::cout << "filter : " << filter << std::endl;

  clFTensor input_tensor(6, 6, 4);
  input_tensor[0] = input;
  input_tensor[1] = input;
  input_tensor[2] = input2;
  input_tensor[3] = input2;
  std::cout << "input : " << input_tensor << std::endl;

  clFTensor output_tensor = layer.compute(input_tensor);

  std::cout << "output : " << output_tensor << std::endl;

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
}

void testMaxPoolingLayer() {
  nnet::CNNMaxPoolingLayer layer({4, 4}, {3, 3});

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
  for (size_t i = 0; i < input_tensor.getDepth(); i++) input_tensor[i] = input;
  for (size_t i = 0; i < input_tensor.getDepth(); i++)
    std::cout << "input " << i << " : \n" << input_tensor[i].toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.compute(input_tensor);

  for (size_t i = 0; i < output_tensor.getDepth(); i++)
    std::cout << "output " << i << " : \n" << output_tensor[i].toFloatMatrix(true) << std::endl;

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
  nnet::CNNMaxPoolingLayer layer({4, 4}, {3, 3});
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
  for (size_t i = 0; i < input_tensor.getDepth(); i++) input_tensor[i] = input;
  for (size_t i = 0; i < input_tensor.getDepth(); i++)
    std::cout << "input " << i << " : \n" << input_tensor[i].toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.computeForward(input_tensor, storage);

  for (size_t i = 0; i < output_tensor.getDepth(); i++)
    std::cout << "output " << i << " : \n" << output_tensor[i].toFloatMatrix(true) << std::endl;


  clFTensor errors_tensor(4, 4, 4);
  for (size_t i = 0; i < errors_tensor.getDepth(); i++) {
    errors_tensor[i].fill(1.f, utils::cl_wrapper.getDefaultQueue(), true);
  }

  clFTensor errors_input = layer.computeBackward(errors_tensor, storage);

  for (size_t i = 0; i < errors_input.getDepth(); i++)
    std::cout << "output " << i << " : \n" << errors_input[i].toFloatMatrix(true) << std::endl;

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
  nnet::CNNAvgPoolingLayer layer({4, 4}, {3, 3});

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
  for (size_t i = 0; i < input_tensor.getDepth(); i++) input_tensor[i] = input;
  for (size_t i = 0; i < input_tensor.getDepth(); i++)
    std::cout << "input " << i << " : \n" << input_tensor[i].toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.compute(input_tensor);

  for (size_t i = 0; i < output_tensor.getDepth(); i++)
    std::cout << "output " << i << " : \n" << output_tensor[i].toFloatMatrix(true) << std::endl;

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
  nnet::CNNAvgPoolingLayer layer({4, 4}, {3, 3});
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
  for (size_t i = 0; i < input_tensor.getDepth(); i++) input_tensor[i] = input;
  for (size_t i = 0; i < input_tensor.getDepth(); i++)
    std::cout << "input " << i << " : \n" << input_tensor[i].toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.computeForward(input_tensor, storage);

  for (size_t i = 0; i < output_tensor.getDepth(); i++)
    std::cout << "output " << i << " : \n" << output_tensor[i].toFloatMatrix(true) << std::endl;

  clFTensor errors_tensor(4, 4, 4);
  for (size_t i = 0; i < errors_tensor.getDepth(); i++) {
    errors_tensor[i].fill(1.f, utils::cl_wrapper.getDefaultQueue(), true);
  }

  clFTensor errors_input = layer.computeBackward(errors_tensor, storage);

  for (size_t i = 0; i < errors_input.getDepth(); i++)
    std::cout << "output " << i << " : \n" << errors_input[i].toFloatMatrix(true) << std::endl;

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
  std::string str_topology("6 6 relu convolution 1 2 2 pooling max 2 2");
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
  for (size_t i = 0; i < filter.getDepth(); i++) filter[i] = f;

  clFTensor input_tensor(6, 6, 3);
  for (size_t i = 0; i < input_tensor.getDepth(); i++) input_tensor[i] = input;
  for (size_t i = 0; i < input_tensor.getDepth(); i++)
    std::cout << "input " << i << " : \n" << input_tensor[i].toFloatMatrix(true) << std::endl;


  clFTensor output_tensor = cnn.predict(input_tensor);

  for (size_t i = 0; i < output_tensor.getDepth(); i++)
    std::cout << "output " << i << " : \n" << output_tensor[i].toFloatMatrix(true) << std::endl;

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
  // std::string str_topology("6 6 relu convolution 2 2 2");
  // std::string str_topology("6 6 relu convolution 2 2 2 pooling max 2 2");
  //std::string str_topology("6 6 relu convolution 2 2 2 convolution 2 2 2");
  std::string str_topology("6 6 relu convolution 2 2 2 convolution 2 2 2 pooling max 2 2");
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
  math::FloatMatrix input2(6, 6);
  {
    input2(0, 0) = 100.f;
    input2(0, 1) = 2.f;
    input2(0, 2) = 1.f;
    input2(0, 3) = 1.f;
    input2(0, 4) = 4.f;
    input2(0, 5) = 1.f;
    input2(1, 0) = 2.f;
    input2(1, 1) = 1.f;
    input2(1, 2) = 1.f;
    input2(1, 3) = 2.f;
    input2(1, 4) = 2.f;
    input2(1, 5) = 1.f;
    input2(2, 0) = 4.f;
    input2(2, 1) = 3.f;
    input2(2, 2) = 2.f;
    input2(2, 3) = 1.f;
    input2(2, 4) = 2.f;
    input2(2, 5) = 1.f;
    input2(3, 0) = 1.f;
    input2(3, 1) = 5.f;
    input2(3, 2) = 1.f;
    input2(3, 3) = 1.f;
    input2(3, 4) = 2.f;
    input2(3, 5) = 1.f;
    input2(4, 0) = 2.f;
    input2(4, 1) = 1.f;
    input2(4, 2) = 1.f;
    input2(4, 3) = 4.f;
    input2(4, 4) = 1.f;
    input2(4, 5) = 1.f;
    input2(5, 0) = 2.f;
    input2(5, 1) = 1.f;
    input2(5, 2) = 4.f;
    input2(5, 3) = 2.f;
    input2(5, 4) = 4.f;
    input2(5, 5) = 1.f;
  }

  const auto &layer1 = dynamic_cast<const nnet::CNNConvolutionLayer *>(cnn.layers[0].get());
  auto &filter1 = layer1->getFilter();
  for (size_t i = 0; i < filter1.getDepth(); i++) filter1[i] = f1;

  const auto &layer2 = dynamic_cast<const nnet::CNNConvolutionLayer *>(cnn.layers[1].get());
  auto &filter2 = layer2->getFilter();
  for (size_t i = 0; i < filter2.getDepth(); i++) filter2[i] = f2;

  clFTensor input_tensor(6, 6, 2);
  input_tensor[0] = input;
  input_tensor[1] = input2;

  std::cout << "input : " << input_tensor << std::endl;

  clFTensor output_tensor = cnn.predict(input_tensor);

  std::cout << "output : " << output_tensor << std::endl;

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
  // testConvolutionalLayer1BranchBP();
  // testConvolutionalLayerXBranch();

  // testMaxPoolingLayer();
  // testMaxPoolingLayerBP();

  // testAvgPoolingLayer();
  // testAvgPoolingLayerBP();

  // testPrediction1Branch();
  testPredictionXBranch();

  return 0;
}