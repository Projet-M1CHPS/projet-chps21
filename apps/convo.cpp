#include "CNN.hpp"
#include "CNNLayer.hpp"
#include "CNNTopology.hpp"
#include "math/Matrix.hpp"
#include "math/clFMatrix.hpp"
#include <iostream>

using namespace math;


void testConvo() {
  cl::CommandQueue queue = utils::cl_wrapper.getDefaultQueue();

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
  std::cout << "input = \n" << A << std::endl;
  auto img = clFTensor(A.getRows(), A.getCols(), 2);
  img[0] = A;
  img[1] = A;


  math::FloatMatrix B1(3, 3);
  {
    B1(0, 0) = 5.f;
    B1(0, 1) = 2.f;
    B1(0, 2) = 2.f;

    B1(1, 0) = 2.f;
    B1(1, 1) = 3.f;
    B1(1, 2) = 1.f;

    B1(2, 0) = 1.f;
    B1(2, 1) = 2.f;
    B1(2, 2) = 2.f;
  }
  math::FloatMatrix B2(3, 3);
  {
    B2(0, 0) = 1.f;
    B2(0, 1) = 1.f;
    B2(0, 2) = 1.f;

    B2(1, 0) = 1.f;
    B2(1, 1) = 1.f;
    B2(1, 2) = 1.f;

    B2(2, 0) = 1.f;
    B2(2, 1) = 1.f;
    B2(2, 2) = 1.f;
  }
  std::cout << "filter 1 = \n" << B1 << std::endl;
  std::cout << "filter 2 = \n" << B2 << std::endl;
  auto filters = clFTensor(B1.getRows(), B1.getCols(), 1);
  filters[0] = B1;

  clFTensor out(4, 4, 2);
  out[0].fill(100.f, queue);
  out[1].fill(100.f, queue);


  clblast::Convgemm<float>(clblast::KernelMode::kCrossCorrelation, 1, 6, 6, 3, 3, 0, 0, 1, 1, 1, 1,
                           filters.getDepth(), img.getDepth(), img.getBuffer()(),
                           img.getOffsetInFloats(), filters.getBuffer()(),
                           filters.getOffsetInFloats(), out.getBuffer()(), out.getOffsetInFloats(),
                           &queue(), nullptr);

  queue.finish();

  std::cout << "output : " << out << std::endl;
}

void testAxpyB() {
  auto queue = utils::cl_wrapper.getDefaultQueue();
  clFTensor tensor(3, 3, 8);

  tensor[0].fill(1.0f, queue);
  tensor[1].fill(2.0f, queue);
  tensor[2].fill(3.0f, queue);
  tensor[3].fill(4.0f, queue);
  tensor[4].fill(5.0f, queue);
  tensor[5].fill(6.0f, queue);
  tensor[6].fill(7.0f, queue);
  tensor[7].fill(8.0f, queue);

  queue.finish();
  std::cout << "before : " << tensor << std::endl;

  clFTensor res(3, 3, 4);
  for (size_t i = 0; i < res.getDepth(); i++) res[i].fill(100.f, queue);

  const size_t n = tensor.getRows() * tensor.getCols();

  std::vector<float> alpha = {1.f, 1.f, 1.f, 1.f};
  std::vector<size_t> x_offset = {0, 18, 36, 54};
  std::vector<size_t> y_offset = {0, 9, 18, 27};
  //{0, 9, 18, 27, 36, 45, 54, 65};
  //{0, 1, 2, 3, 4, 5, 6, 7};

  clblast::AxpyBatched<float>(n, alpha.data(), tensor.getBuffer()(), x_offset.data(), 1,
                              res.getBuffer()(), y_offset.data(), 1, 4, &queue(), nullptr);

  std::vector<size_t> xx_offset = {9, 27, 45, 63};

  clblast::AxpyBatched<float>(n, alpha.data(), tensor.getBuffer()(), xx_offset.data(), 1,
                              res.getBuffer()(), y_offset.data(), 1, 4, &queue(), nullptr);

  queue.finish();
  std::cout << "after : " << res << std::endl;
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

  clFTensor output_tensor = layer.compute(utils::cl_wrapper.getDefaultQueue(), input_tensor);

  utils::cl_wrapper.getDefaultQueue().finish();

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
  const size_t number_input = 6;
  const size_t number_filter = 2;
  const size_t number_branch = 3;

  nnet::CNNConvolutionLayer layer({5, 5}, {2, 2}, number_filter, af::ActivationFunctionType::relu,
                                  number_branch);

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
  filter[1] = f2;
  filter[2] = f1;
  filter[3] = f2;
  filter[4] = f1;
  filter[5] = f2;

  std::cout << "filter : " << filter << std::endl;

  math::FloatMatrix input1(6, 6);
  {
    input1(0, 0) = 1.f;
    input1(0, 1) = 2.f;
    input1(0, 2) = 1.f;
    input1(0, 3) = 1.f;
    input1(0, 4) = 4.f;
    input1(0, 5) = 1.f;
    input1(1, 0) = 2.f;
    input1(1, 1) = 1.f;
    input1(1, 2) = 1.f;
    input1(1, 3) = 2.f;
    input1(1, 4) = 2.f;
    input1(1, 5) = 1.f;
    input1(2, 0) = 4.f;
    input1(2, 1) = 3.f;
    input1(2, 2) = 2.f;
    input1(2, 3) = 1.f;
    input1(2, 4) = 2.f;
    input1(2, 5) = 1.f;
    input1(3, 0) = 1.f;
    input1(3, 1) = 5.f;
    input1(3, 2) = 1.f;
    input1(3, 3) = 1.f;
    input1(3, 4) = 2.f;
    input1(3, 5) = 1.f;
    input1(4, 0) = 2.f;
    input1(4, 1) = 1.f;
    input1(4, 2) = 1.f;
    input1(4, 3) = 4.f;
    input1(4, 4) = 1.f;
    input1(4, 5) = 1.f;
    input1(5, 0) = 2.f;
    input1(5, 1) = 1.f;
    input1(5, 2) = 4.f;
    input1(5, 3) = 2.f;
    input1(5, 4) = 4.f;
    input1(5, 5) = 1.f;
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

  clFTensor input_tensor(6, 6, number_input);
  input_tensor[0] = input1;
  input_tensor[1] = input2;
  input_tensor[2] = input1;
  input_tensor[3] = input2;
  input_tensor[4] = input1;
  input_tensor[5] = input2;
  std::cout << "input : " << input_tensor << std::endl;

  clFTensor output_tensor = layer.computeForward(utils::cl_wrapper.getDefaultQueue(), input_tensor, storage);

  utils::cl_wrapper.getDefaultQueue().finish();

  std::cout << "output : " << output_tensor << std::endl;

  std::cout << "input storage : " << storage.input << std::endl;

  clFTensor errors_tensor(5, 5, number_filter * number_branch * (number_input / number_branch));
  errors_tensor[0].fill(1.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[1].fill(2.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[2].fill(3.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[3].fill(4.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[4].fill(5.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[5].fill(6.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[6].fill(7.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[7].fill(8.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[8].fill(9.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[9].fill(10.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[10].fill(11.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[11].fill(12.f, utils::cl_wrapper.getDefaultQueue(), true);

  std::cout << "error : " << errors_tensor << std::endl;

  clFTensor errors_input = layer.computeBackward(utils::cl_wrapper.getDefaultQueue(), errors_tensor, storage);

  utils::cl_wrapper.getDefaultQueue().finish();

  // std::cout << "error filter : " << storage.errorFilter << std::endl;
  std::cout << "error input : " << errors_input << std::endl;

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
  math::FloatMatrix input2(input);
  input2(0, 0) = 100.f;

  auto &filter = layer.getFilter();
  filter[0] = f1;
  filter[1] = f2;
  filter[2] = f3;
  filter[3] = f4;

  std::cout << "filter : " << filter << std::endl;

  clFTensor input_tensor(6, 6, 4);
  input_tensor[0] = input;
  input_tensor[1] = input2;
  input_tensor[2] = input;
  input_tensor[3] = input2;
  std::cout << "input : " << input_tensor << std::endl;

  clFTensor output_tensor = layer.compute(utils::cl_wrapper.getDefaultQueue(), input_tensor);

  utils::cl_wrapper.getDefaultQueue().finish();

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

  clFTensor output_tensor = layer.compute(utils::cl_wrapper.getDefaultQueue(), input_tensor);

  utils::cl_wrapper.getDefaultQueue().finish();

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

  clFTensor output_tensor = layer.computeForward(utils::cl_wrapper.getDefaultQueue(), input_tensor, storage);

  utils::cl_wrapper.getDefaultQueue().finish();

  for (size_t i = 0; i < output_tensor.getDepth(); i++)
    std::cout << "output " << i << " : \n" << output_tensor[i].toFloatMatrix(true) << std::endl;


  clFTensor errors_tensor(4, 4, 4);
  for (size_t i = 0; i < errors_tensor.getDepth(); i++) {
    errors_tensor[i].fill(1.f, utils::cl_wrapper.getDefaultQueue(), true);
  }

  clFTensor errors_input = layer.computeBackward(utils::cl_wrapper.getDefaultQueue(), errors_tensor, storage);

  utils::cl_wrapper.getDefaultQueue().finish();

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

  math::FloatMatrix input1(6, 6);
  {
    input1(0, 0) = 1.f;
    input1(0, 1) = 2.f;
    input1(0, 2) = 1.f;
    input1(0, 3) = 1.f;
    input1(0, 4) = 4.f;
    input1(0, 5) = 1.f;
    input1(1, 0) = 2.f;
    input1(1, 1) = 1.f;
    input1(1, 2) = 1.f;
    input1(1, 3) = 2.f;
    input1(1, 4) = 2.f;
    input1(1, 5) = 1.f;
    input1(2, 0) = 4.f;
    input1(2, 1) = 3.f;
    input1(2, 2) = 2.f;
    input1(2, 3) = 1.f;
    input1(2, 4) = 2.f;
    input1(2, 5) = 1.f;
    input1(3, 0) = 1.f;
    input1(3, 1) = 5.f;
    input1(3, 2) = 1.f;
    input1(3, 3) = 1.f;
    input1(3, 4) = 2.f;
    input1(3, 5) = 1.f;
    input1(4, 0) = 2.f;
    input1(4, 1) = 1.f;
    input1(4, 2) = 1.f;
    input1(4, 3) = 4.f;
    input1(4, 4) = 1.f;
    input1(4, 5) = 1.f;
    input1(5, 0) = 2.f;
    input1(5, 1) = 1.f;
    input1(5, 2) = 4.f;
    input1(5, 3) = 2.f;
    input1(5, 4) = 4.f;
    input1(5, 5) = 1.f;
  }
  math::FloatMatrix input2(input1);
  input2(0, 0) = 100.f;

  clFTensor input_tensor(6, 6, 2);
  input_tensor[0] = input1;
  input_tensor[1] = input2;
  for (size_t i = 0; i < input_tensor.getDepth(); i++)
    std::cout << "input " << i << " : \n" << input_tensor[i].toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.compute(utils::cl_wrapper.getDefaultQueue(), input_tensor);

  utils::cl_wrapper.getDefaultQueue().finish();

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
  math::FloatMatrix input2(input);
  input2(0, 0) = 100.f;

  clFTensor input_tensor(6, 6, 2);
  input_tensor[0] = input;
  input_tensor[1] = input2;
  for (size_t i = 0; i < input_tensor.getDepth(); i++)
    std::cout << "input " << i << " : \n" << input_tensor[i].toFloatMatrix(true) << std::endl;

  clFTensor output_tensor = layer.computeForward(utils::cl_wrapper.getDefaultQueue(), input_tensor, storage);

  utils::cl_wrapper.getDefaultQueue().finish();

  for (size_t i = 0; i < output_tensor.getDepth(); i++)
    std::cout << "output " << i << " : \n" << output_tensor[i].toFloatMatrix(true) << std::endl;

  clFTensor errors_tensor(4, 4, 2);
  for (size_t i = 0; i < errors_tensor.getDepth(); i++) {
    errors_tensor[i].fill(static_cast<float>(i + 1), utils::cl_wrapper.getDefaultQueue(), true);
  }

  clFTensor errors_input = layer.computeBackward(utils::cl_wrapper.getDefaultQueue(), errors_tensor, storage);

  utils::cl_wrapper.getDefaultQueue().finish();

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

  const auto &layer = cnn.getLayers()[0].get();
  const auto &layer_convolution = dynamic_cast<const nnet::CNNConvolutionLayer *>(layer);
  auto &filter = layer_convolution->getFilter();
  for (size_t i = 0; i < filter.getDepth(); i++) filter[i] = f;

  clFTensor input_tensor(6, 6, 3);
  for (size_t i = 0; i < input_tensor.getDepth(); i++) input_tensor[i] = input;
  for (size_t i = 0; i < input_tensor.getDepth(); i++)
    std::cout << "input " << i << " : \n" << input_tensor[i].toFloatMatrix(true) << std::endl;


  clFTensor output_tensor = cnn.predict(utils::cl_wrapper.getDefaultQueue(), input_tensor);

  utils::cl_wrapper.getDefaultQueue().finish();

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
  // std::string str_topology("6 6 relu convolution 2 2 2 convolution 2 2 2");
  std::string str_topology("6 6 relu convolution 2 2 2 convolution 2 2 2 pooling avg 2 2");
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

  const auto &layer1 = dynamic_cast<const nnet::CNNConvolutionLayer *>(cnn.getLayers()[0].get());
  auto &filter1 = layer1->getFilter();
  filter1[0] = f1;
  filter1[1] = f2;

  const auto &layer2 = dynamic_cast<const nnet::CNNConvolutionLayer *>(cnn.getLayers()[1].get());
  auto &filter2 = layer2->getFilter();
  filter2[0] = f1;
  filter2[1] = f2;
  filter2[2] = f1;
  filter2[3] = f2;

  clFTensor input_tensor(6, 6, 2);
  input_tensor[0] = input;
  input_tensor[1] = input2;

  std::cout << "input : " << input_tensor << std::endl;

  clFTensor output_tensor = cnn.predict(utils::cl_wrapper.getDefaultQueue(), input_tensor);

  utils::cl_wrapper.getDefaultQueue().finish();

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


void foo() {
  // 1 branch 2 filter/branch 2 input
  const size_t number_input = 4;
  const size_t number_filter = 2;
  const size_t number_branch = 2;

  nnet::CNNConvolutionLayer layer({5, 5}, {2, 2}, number_filter, af::ActivationFunctionType::relu,
                                  number_branch);

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
  auto &filter = layer.getFilter();
  filter[0] = f1;
  filter[1] = f2;
  filter[2] = f3;
  filter[3] = f4;


  std::cout << "filter : " << filter << std::endl;

  math::FloatMatrix input1(6, 6);
  {
    input1(0, 0) = 1.f;
    input1(0, 1) = 2.f;
    input1(0, 2) = 1.f;
    input1(0, 3) = 1.f;
    input1(0, 4) = 4.f;
    input1(0, 5) = 1.f;
    input1(1, 0) = 2.f;
    input1(1, 1) = 1.f;
    input1(1, 2) = 1.f;
    input1(1, 3) = 2.f;
    input1(1, 4) = 2.f;
    input1(1, 5) = 1.f;
    input1(2, 0) = 4.f;
    input1(2, 1) = 3.f;
    input1(2, 2) = 2.f;
    input1(2, 3) = 1.f;
    input1(2, 4) = 2.f;
    input1(2, 5) = 1.f;
    input1(3, 0) = 1.f;
    input1(3, 1) = 5.f;
    input1(3, 2) = 1.f;
    input1(3, 3) = 1.f;
    input1(3, 4) = 2.f;
    input1(3, 5) = 1.f;
    input1(4, 0) = 2.f;
    input1(4, 1) = 1.f;
    input1(4, 2) = 1.f;
    input1(4, 3) = 4.f;
    input1(4, 4) = 1.f;
    input1(4, 5) = 1.f;
    input1(5, 0) = 2.f;
    input1(5, 1) = 1.f;
    input1(5, 2) = 4.f;
    input1(5, 3) = 2.f;
    input1(5, 4) = 4.f;
    input1(5, 5) = 1.f;
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

  clFTensor input_tensor(6, 6, number_input);
  input_tensor[0] = input1;
  input_tensor[1] = input2;
  input_tensor[2] = input1;
  input_tensor[3] = input2;
  std::cout << "input : " << input_tensor << std::endl;

  clFTensor output_tensor = layer.computeForward(utils::cl_wrapper.getDefaultQueue(), input_tensor, storage);

  utils::cl_wrapper.getDefaultQueue().finish();

  std::cout << "output : " << output_tensor << std::endl;

  std::cout << "input storage : " << storage.input << std::endl;

  clFTensor errors_tensor(5, 5, number_filter * number_branch * (number_input / number_branch));
  errors_tensor[0].fill(1.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[1].fill(2.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[2].fill(3.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[3].fill(4.f, utils::cl_wrapper.getDefaultQueue(), true);

  errors_tensor[4].fill(5.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[5].fill(6.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[6].fill(7.f, utils::cl_wrapper.getDefaultQueue(), true);
  errors_tensor[7].fill(8.f, utils::cl_wrapper.getDefaultQueue(), true);

  std::cout << "error : " << errors_tensor << std::endl;

  clFTensor errors_input = layer.computeBackward(utils::cl_wrapper.getDefaultQueue(), errors_tensor, storage);

  utils::cl_wrapper.getDefaultQueue().finish();

  std::cout << "error input : " << errors_input << std::endl;
  std::cout << "error filter : " << storage.error_filter << std::endl;

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

int main() {
  utils::clWrapper::initOpenCL(*utils::clWrapper::makeDefault());

  // testConvo();
  // testAxpyB();

  // testConvolutionalLayer1Branch();
  // testConvolutionalLayer1BranchBP();
  foo();
  // testConvolutionalLayerXBranch();

  // testMaxPoolingLayer();
  // testMaxPoolingLayerBP();

  // testAvgPoolingLayer();
  //testAvgPoolingLayerBP();

  // testPrediction1Branch();
  // testPredictionXBranch();

  return 0;
}