#include "ActivationFunction.hpp"
#include "NeuralNetwork.hpp"
#include "Utils.hpp"
#include "controlSystem/classifier/classifierController.hpp"
#include <iomanip>
#include <iostream>
#include <vector>

template<typename T>
size_t func_xor(const size_t bach_size, const T learning_rate, const T error_limit) {
  nnet::NeuralNetwork<T> nn;
  nn.setLayersSize(std::vector<size_t>{2, 3, 3, 1});
  nn.setActivationFunction(af::ActivationFunctionType::sigmoid);
  // nn.setActivationFunction(af::ActivationFunctionType::sigmoid, 2);
  nn.randomizeSynapses();

  std::cout << nn << std::endl;

  std::vector<std::vector<T>> input{{1, 1}, {1, 0}, {0, 1}, {0, 0}};
  std::vector<T> target{0, 1, 1, 0};

  T error = 1.0;
  size_t count = 0;
  while (error > error_limit) {
    for (int i = 0; i < bach_size; i++) {
      for (int j = 0; j < 4; j++) {
        nn.train(input[j].begin(), input[j].end(), target.begin() + j, target.begin() + j + 1,
                 learning_rate);
      }
    }

    error = 0.0;
    for (int i = 0; i < input.size(); i++)
      error += std::fabs(nn.predict(input[i].begin(), input[i].end())(0, 0) - target[i]);
    error /= input.size();
    std::cout << std::setprecision(17) << error << std::endl;
    count++;
  }

  std::cout << nn << std::endl;
  std::cout << "Result"
            << "---> " << count << " iterations" << std::endl;
  for (int i = 0; i < input.size(); i++) {
    std::cout << input[i][0] << "|" << input[i][1] << " = "
              << nn.predict(input[i].begin(), input[i].end()) << "(" << target[i] << ")"
              << std::endl;
  }
  return count;
}

void test_neural_network() {
  nnet::NeuralNetwork<float> nn;
  nn.setLayersSize(std::vector<size_t>{2, 2, 2});
  nn.setActivationFunction(af::ActivationFunctionType::sigmoid);

  math::Matrix<float> &w1 = nn.getWeights()[0];
  math::Matrix<float> &b1 = nn.getBiases()[0];
  math::Matrix<float> &w2 = nn.getWeights()[1];
  math::Matrix<float> &b2 = nn.getBiases()[1];

  w1(0, 0) = 0.15;   // w1
  w1(0, 1) = 0.20;   // w3
  w1(1, 0) = 0.25;   // w2
  w1(1, 1) = 0.30;   // w4
  b1(0, 0) = 0.35;   // b1
  b1(1, 0) = 0.35;   // b2

  w2(0, 0) = 0.40;   // w1
  w2(0, 1) = 0.45;   // w3
  w2(1, 0) = 0.50;   // w2
  w2(1, 1) = 0.55;   // w4
  b2(0, 0) = 0.60;   // b1
  b2(1, 0) = 0.60;   // b2

  std::cout << nn << std::endl;

  math::Matrix<float> input = {0.05, 0.10};
  math::Matrix<float> output = {0.01, 0.99};

  std::cout << "prediction : \n" << nn.predict(input) << std::endl;

  nn.train(input, output, 0.5);
  std::cout << nn << std::endl;
}

using namespace control;
using namespace control::classifier;

bool test_image() {
  std::filesystem::path input_path = "/home/thukisdo/Bureau/truncated_testing_set";

  auto loader = std::make_shared<CITCLoader>(16, 16);
  auto &engine = loader->getPostProcessEngine();
  engine.addTransformation(std::make_shared<image::transform::BinaryScaleByMedian>());

  CTParams parameters(RunPolicy::create, input_path, nullptr, "runs/test");
  parameters.setTrainingSetLoader<CITCLoader>(16, 16);


  std::vector<size_t> topology = {16 * 16, 64, 32, 8, 2};
  parameters.setTopology(topology.begin(), topology.end());

  CTController controller(parameters);
  ControllerResult res = controller.run(true, &std::cout);


  if (not res) { res.print(std::cout); }

  return (bool) res;
}

int main(int argc, char **argv) {
  // func_xor<float>(100, 0.2, 0.001);
  //  test();
  //  test_neural_network();
  return test_image();
}