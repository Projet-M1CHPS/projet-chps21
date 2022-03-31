
#include "NeuralNetwork.hpp"
#include "ProjectVersion.hpp"
#include "tscl.hpp"

#include "CNN.hpp"
#include "CNNLayer.hpp"
#include "CNNModel.hpp"
#include "CNNStorageBP.hpp"
#include <curses.h>
#include <iomanip>
#include <iostream>
#include <vector>


void runXor(const size_t bach_size, const float learning_rate, const float error_limit) {
  /*nnet::MLPModel model;
  auto &nn1 = model.getPerceptron();
  nnet::MLPTopology topology = {2, 3, 3, 1};
  nn1.setTopology(topology);
  nn1.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn1.randomizeWeight();

  auto &w1 = nn1.getWeights();

  auto tmStandard = std::make_unique<nnet::SGDOptimization>(nn1, 0.2f);

  nnet::MLPStochOptimizer opt1(model, tmStandard);

  std::vector<math::FloatMatrix> input_buffer(4);
  std::vector<math::FloatMatrix> target_buffer(4);
  for (size_t i = 0; i < 4; i++) {
    input_buffer[i] = math::FloatMatrix(2, 1);
    target_buffer[i] = math::FloatMatrix(1, 1);
  }

  // Xor truth table
  // Yes, this is ugly, but who cares ?
  input_buffer[0](0, 0) = 0.f;
  input_buffer[0](1, 0) = 0.f;
  input_buffer[1](0, 0) = 1.f;
  input_buffer[1](1, 0) = 0.f;
  input_buffer[2](0, 0) = 0.f;
  input_buffer[2](1, 0) = 1.f;
  input_buffer[3](0, 0) = 1.f;
  input_buffer[3](1, 0) = 1.f;

  target_buffer[0](0, 0) = 0.f;
  target_buffer[1](0, 0) = 1.f;
  target_buffer[2](0, 0) = 1.f;
  target_buffer[3](0, 0) = 0.f;

  std::vector<math::clFMatrix> input(4);
  std::vector<math::clFMatrix> target(4);
  for (size_t i = 0; i < 4; i++) {
    input[i] = input_buffer[i];
    target[i] = target_buffer[i];
  }

  float error = 1.0;
  size_t count = 0;
  std::cout << std::setprecision(8);
  while (error > error_limit && count < 600) {
    for (int i = 0; i < bach_size; i++) {
      for (int j = 0; j < 4; j++) { opt1.optimize(input[j], target[j]); }
    }

    error = 0.0;
    for (int i = 0; i < input.size(); i++) {
      error += std::fabs(nn1.predict(input[i])(0, 0) - target[i](0, 0));
    }
    error /= (float)input.size();
    std::cout << error << std::endl;
    count++;
  }

  std::cout << nn1 << std::endl;
  std::cout << "Result"
            << "---> " << count << " iterations" << std::endl;
  for (int i = 0; i < input.size(); i++) {
    std::cout << input[i](0, 0) << "|" << input[i](1, 0) << " = " << nn1.predict(input[i]) << "("
              << target[i](0, 0) << ")" << std::endl;
  }*/
}


void testConvolutionalLayer() {
  nnet::CNNConvolutionLayer layer({2, 2}, af::ActivationFunctionType::relu, 1, 0);

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

  math::clFMatrix output(5, 5);

  layer.compute(input, output);

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

  nnet::CNNStorageBPConvolution storage({6, 6}, {5, 5}, {2, 2}, 1);

  layer.computeForward(input, storage);
  std::cout << "compute forward : \n" << storage.output << std::endl;
  for (auto &i : storage.output) { i = 1.f; }
  layer.computeBackward(input, storage);
  std::cout << "error input : \n" << storage.errorInput << std::endl;
  std::cout << "error filter : \n" << storage.errorFilter << std::endl;

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
  nnet::CNNMaxPoolingLayer layer({3, 3}, 1);

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

  math::clFMatrix output(4, 4);

  layer.compute(input, output);

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

  nnet::CNNStorageBPMaxPooling storage({6, 6}, {4, 4});

  layer.computeForward(input, storage);
  std::cout << "compute forward : \n" << storage.output << std::endl;
  for (auto &i : storage.output) { i = 1.f; }
  for (auto &i : storage.errorInput) { i = 0.f; }
  layer.computeBackward(input, storage);
  std::cout << "error input : \n" << storage.errorInput << std::endl;

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
  nnet::CNNAvgPoolingLayer layer({3, 3}, 1);

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

  math::clFMatrix output(4, 4);

  layer.compute(input, output);

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

  nnet::CNNStorageBPAvgPooling storage({6, 6}, {4, 4});

  layer.computeForward(input, storage);
  std::cout << "compute forward : \n" << storage.output << std::endl;
  for (auto &i : storage.output) { i = 1.f; }
  layer.computeBackward(input, storage);
  std::cout << "error input : \n" << storage.errorInput << std::endl;

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

  math::clFMatrix tmp_input = input;
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

void testPredictionMultiBranch() {
  std::string str_topo("6 6 relu convolution 2 2 2 1 0 pooling max 2 2 1");
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
  math::clFMatrix output(32, 1);

  std::cout << input << std::endl;

  cnn.predict(input, output);

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

void testCNNModel() {
  std::string str_topo("6 6 relu convolution 1 2 2 1 0 pooling max 2 2 1");
  auto topo_cnn = nnet::stringToTopology(str_topo);
  std::cout << topo_cnn << std::endl;
  nnet::MLPTopology topo_mlp = {5, 2};

  std::unique_ptr<nnet::CNNModel> model = nnet::CNNModel::random(topo_cnn, topo_mlp);

  math::FloatMatrix input_buffer(6, 6);
  {
    input_buffer(0, 0) = 1.f;
    input_buffer(0, 1) = 2.f;
    input_buffer(0, 2) = 1.f;
    input_buffer(0, 3) = 1.f;
    input_buffer(0, 4) = 4.f;
    input_buffer(0, 5) = 1.f;
    input_buffer(1, 0) = 2.f;
    input_buffer(1, 1) = 1.f;
    input_buffer(1, 2) = 1.f;
    input_buffer(1, 3) = 2.f;
    input_buffer(1, 4) = 2.f;
    input_buffer(1, 5) = 1.f;
    input_buffer(2, 0) = 4.f;
    input_buffer(2, 1) = 3.f;
    input_buffer(2, 2) = 2.f;
    input_buffer(2, 3) = 1.f;
    input_buffer(2, 4) = 2.f;
    input_buffer(2, 5) = 1.f;
    input_buffer(3, 0) = 1.f;
    input_buffer(3, 1) = 5.f;
    input_buffer(3, 2) = 1.f;
    input_buffer(3, 3) = 1.f;
    input_buffer(3, 4) = 2.f;
    input_buffer(3, 5) = 1.f;
    input_buffer(4, 0) = 2.f;
    input_buffer(4, 1) = 1.f;
    input_buffer(4, 2) = 1.f;
    input_buffer(4, 3) = 4.f;
    input_buffer(4, 4) = 1.f;
    input_buffer(4, 5) = 1.f;
    input_buffer(5, 0) = 2.f;
    input_buffer(5, 1) = 1.f;
    input_buffer(5, 2) = 4.f;
    input_buffer(5, 3) = 2.f;
    input_buffer(5, 4) = 4.f;
    input_buffer(5, 5) = 1.f;
  }
  std::cout << "input :\n" << input_buffer << std::endl;

  math::clFMatrix input = input_buffer;
  math::clFMatrix output = model->predict(input);

  math::FloatMatrix output_buffer = output.toFloatMatrix(true);
  std::cout << "output :\n" << output_buffer << std::endl;
}

int main() {
  // runXor(100, 0.1, 0.001);

  // testConvolutionalLayer();
  // testMaxPoolingLayer();
  // testAvgPoolingLayer();
  testPredictionOneBranch();
  // testPredictionMultiBranch();
  // testCNNModel();

  return 0;
}