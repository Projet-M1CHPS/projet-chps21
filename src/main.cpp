#include "Control.hpp"
#include "Network.hpp"
#include "ProjectVersion.hpp"
#include "tscl.hpp"

#include <iomanip>
#include <iostream>
#include <vector>


// using namespace control;

template<typename T>
size_t func_xor(const size_t bach_size, const T learning_rate, const T error_limit) {
  std::vector<size_t> topology{2, 3, 3, 1};

  nnet::MLPerceptron<T> nn;
  nn.setTopology(topology);
  nn.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn.randomizeWeight();

  nnet::SGDOptimization<T> std_mt(learning_rate);
  nnet::MomentumOptimization<T> mom_mt(topology, learning_rate, 0.9);
  nnet::DecayOptimization<T> decay_mt(learning_rate, 0.1f);
  nnet::DecayMomentumOptimization<T> momDecay_mt(topology, learning_rate, 0.01f, 0.9f);
  nnet::RPropPOptimization<T> rprop(topology);

  nnet::MLPModelStochOptimizer<T> opt(&nn, &decay_mt);

  std::cout << nn << std::endl;

  std::vector<math::Matrix<T>> input(4);
  std::vector<math::Matrix<T>> target(4);

  for (size_t i = 0; i < 4; i++) {
    input[i] = math::Matrix<T>(2, 1);
    target[i] = math::Matrix<T>(1, 1);
  }
  input[0](0, 0) = 0.f;
  input[0](1, 0) = 0.f;
  input[1](0, 0) = 1.f;
  input[1](1, 0) = 0.f;
  input[2](0, 0) = 1.f;
  input[2](1, 0) = 1.f;
  input[3](0, 0) = 0.f;
  input[3](1, 0) = 1.f;


  target[0](0, 0) = 0.f;
  target[1](0, 0) = 1.f;
  target[2](0, 0) = 0.f;
  target[3](0, 0) = 1.f;

  // for (int j = 0; j < 4; j++) { opt.train(input[j], target[j]); }

  T error = 1.0;
  size_t count = 0;
  while (error > error_limit && count < 1000) {
    for (int i = 0; i < bach_size; i++) { opt.train(input.begin(), input.end(), target.begin()); }

    error = 0.0;
    for (int i = 0; i < input.size(); i++)
      error += std::fabs(nn.predict(input[i])(0, 0) - target[i](0, 0));
    error /= input.size();
    // std::cout << count << " " << std::setprecision(17) << error << std::endl;
    count++;
    momDecay_mt.update();
    decay_mt.update();
  }

  std::cout << nn << std::endl;
  std::cout << "Result"
            << "---> " << count << " iterations" << std::endl;
  for (int i = 0; i < input.size(); i++) {
    std::cout << input[i](0, 0) << "|" << input[i](1, 0) << " = " << nn.predict(input[i]) << "("
              << target[i] << ")" << std::endl;
  }
  return count;
}


template<typename T>
size_t func_xor_batch(const size_t bach_size, const T learning_rate, const T error_limit) {
  std::vector<size_t> topology{2, 3, 3, 1};

  nnet::MLPerceptron<T> nn;
  nn.setTopology(topology);
  nn.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn.randomizeWeight();

  nnet::SGDOptimization<T> std_mt(learning_rate);
  nnet::MomentumOptimization<T> mom_mt(topology, learning_rate, 0.8);
  nnet::DecayOptimization<T> decay_mt(learning_rate, 0.01f);
  nnet::DecayMomentumOptimization<T> momDecay_mt(topology, learning_rate, 0.01f, 0.8f);
  nnet::RPropPOptimization<T> rprop_mt(topology);

  nnet::MLPBatchOptimizer<T> opt(&nn, &momDecay_mt);

  std::cout << nn << std::endl;

  std::vector<math::Matrix<T>> input(4);
  std::vector<math::Matrix<T>> target(4);

  for (size_t i = 0; i < 4; i++) {
    input[i] = math::Matrix<T>(2, 1);
    target[i] = math::Matrix<T>(1, 1);
  }
  input[0](0, 0) = 0.f;
  input[0](1, 0) = 0.f;
  input[1](0, 0) = 1.f;
  input[1](1, 0) = 0.f;
  input[2](0, 0) = 1.f;
  input[2](1, 0) = 1.f;
  input[3](0, 0) = 0.f;
  input[3](1, 0) = 1.f;


  target[0](0, 0) = 0.f;
  target[1](0, 0) = 1.f;
  target[2](0, 0) = 0.f;
  target[3](0, 0) = 1.f;

  T error = 1.0;
  size_t count = 0;
  while (error > error_limit && count < 1000) {
    for (int i = 0; i < bach_size; i++) { opt.train(input.begin(), input.end(), target.begin()); }

    error = 0.0;
    for (int i = 0; i < input.size(); i++)
      error += std::fabs(nn.predict(input[i])(0, 0) - target[i](0, 0));
    error /= input.size();
    std::cout << count << " " << std::setprecision(17) << error << std::endl;
    count++;
    momDecay_mt.update();
    decay_mt.update();
  }

  std::cout << nn << std::endl;
  std::cout << "Result"
            << "---> " << count << " iterations" << std::endl;
  for (int i = 0; i < input.size(); i++) {
    std::cout << input[i](0, 0) << "|" << input[i](1, 0) << " = " << nn.predict(input[i]) << "("
              << target[i] << ")" << std::endl;
  }

  return count;
}


template<typename T>
void new_func_xor(const size_t bach_size, const T learning_rate, const T error_limit) {
  nnet::MLPerceptron<T> nn1;
  std::vector<size_t> topology = {2, 3, 3, 1};
  nn1.setTopology(topology);
  nn1.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn1.randomizeWeight();

  nnet::MLPerceptron<T> nn2(nn1);
  nn2.setTopology(topology);
  nn2.setActivationFunction(af::ActivationFunctionType::sigmoid);

  auto &w1 = nn1.getWeights();
  auto &w2 = nn2.getWeights();
  for (size_t i = 0; i < w1.size(); i++) w2[i] = w1[i];

  nnet::SGDOptimization<T> tmStandard(0.2f);
  nnet::MomentumOptimization<T> tmMomentum(topology, 0.1f, 0.9f);

  nnet::MLPModelStochOptimizer<T> opt1(&nn1, &tmStandard);
  nnet::MLPModelStochOptimizer<T> opt2(&nn2, &tmMomentum);

  std::cout << nn1 << std::endl;
  std::cout << nn2 << std::endl;

  std::vector<math::Matrix<T>> input(4);
  std::vector<math::Matrix<T>> target(4);
  for (size_t i = 0; i < 4; i++) {
    input[i] = math::Matrix<T>(2, 1);
    target[i] = math::Matrix<T>(1, 1);
  }
  input[0](0, 0) = 0.f;
  input[0](1, 0) = 0.f;
  input[1](0, 0) = 1.f;
  input[1](1, 0) = 0.f;
  input[2](0, 0) = 0.f;
  input[2](1, 0) = 1.f;
  input[3](0, 0) = 1.f;
  input[3](1, 0) = 1.f;

  target[0](0, 0) = 0.f;
  target[1](0, 0) = 1.f;
  target[2](0, 0) = 1.f;
  target[3](0, 0) = 0.f;

  for (int z = 0; z < 2; z++) {
    T error = 1.0;
    size_t count = 0;
    while (error > error_limit && count < 600) {
      for (int i = 0; i < bach_size; i++) {
        for (int j = 0; j < 4; j++) {
          if (z == 0) {
            opt1.train(input[j], target[j]);
          } else {
            opt2.train(input[j], target[j]);
          }
        }
      }

      error = 0.0;
      for (int i = 0; i < input.size(); i++) {
        if (z == 0) error += std::fabs(nn1.predict(input[i])(0, 0) - target[i](0, 0));
        else
          error += std::fabs(nn2.predict(input[i])(0, 0) - target[i](0, 0));
      }
      error /= input.size();
      std::cout << std::setprecision(17) << error << std::endl;
      count++;
    }

    if (z == 0) {
      std::cout << nn1 << std::endl;
      std::cout << "Result"
                << "---> " << count << " iterations" << std::endl;
      for (int i = 0; i < input.size(); i++) {
        std::cout << input[i](0, 0) << "|" << input[i](1, 0) << " = " << nn1.predict(input[i])
                  << "(" << target[i](0, 0) << ")" << std::endl;
      }
    } else {
      std::cout << nn2 << std::endl;
      std::cout << "Result"
                << "---> " << count << " iterations" << std::endl;
      for (int i = 0; i < input.size(); i++) {
        std::cout << input[i](0, 0) << "|" << input[i](1, 0) << " = " << nn2.predict(input[i])
                  << "(" << target[i](0, 0) << ")" << std::endl;
      }
    }
  }
}


template<typename T>
void batch(const size_t bach_size, const T learning_rate, const T error_limit) {
  nnet::MLPerceptron<T> nn1;
  std::vector<size_t> topology = {2, 3, 3, 1};
  nn1.setTopology(topology);
  nn1.setActivationFunction(af::ActivationFunctionType::sigmoid);
  nn1.randomizeWeight();

  nnet::MLPerceptron<T> nn2(nn1);
  nn2.setTopology(topology);
  nn2.setActivationFunction(af::ActivationFunctionType::sigmoid);

  auto &w1 = nn1.getWeights();
  auto &w2 = nn2.getWeights();
  for (size_t i = 0; i < w1.size(); i++) w2[i] = w1[i];

  auto &b1 = nn1.getBiases();
  auto &b2 = nn2.getBiases();
  for (size_t i = 0; i < w1.size(); i++) b2[i] = b1[i];

  nnet::SGDOptimization<T> tmStandard1(0.2f);
  nnet::SGDOptimization<T> tmStandard2(0.2f);

  nnet::MLPModelStochOptimizer<T> opt1(&nn1, &tmStandard1);
  nnet::MLPBatchOptimizer<T> opt2(&nn2, &tmStandard2);

  std::cout << nn1 << std::endl;
  std::cout << nn2 << std::endl;

  std::vector<math::Matrix<T>> input(4);
  std::vector<math::Matrix<T>> target(4);
  for (size_t i = 0; i < 4; i++) {
    input[i] = math::Matrix<T>(2, 1);
    target[i] = math::Matrix<T>(1, 1);
  }
  input[0](0, 0) = 0.f;
  input[0](1, 0) = 0.f;
  input[1](0, 0) = 1.f;
  input[1](1, 0) = 0.f;
  input[2](0, 0) = 0.f;
  input[2](1, 0) = 1.f;
  input[3](0, 0) = 1.f;
  input[3](1, 0) = 1.f;

  target[0](0, 0) = 0.f;
  target[1](0, 0) = 1.f;
  target[2](0, 0) = 1.f;
  target[3](0, 0) = 0.f;


  for (int j = 0; j < 4; j++) { opt1.train(input[j], target[j]); }

  std::cout << "\n\n---------------------------------\n\n" << std::endl;

  opt2.train(input, target);
}


using namespace control;
using namespace control::classifier;
using namespace tscl;

void setupLogger() {
  auto &handler = logger.addHandler<StreamLogHandler>("term", std::cout);
  handler.tsType(tscl::timestamp_t::Partial);
  handler.minLvl(Log::Trace);
  // handler.enable(false);

  auto &thandler = logger.addHandler<StreamLogHandler>("file", "logs.txt");
  thandler.minLvl(Log::Information);
}

bool test_image(std::vector<std::string> const &args) {
  // FIXME: placeholder path
  setupLogger();

  if (args.size() < 2) {
    tscl::logger("Usage: " + args[0] + " <input_path> (<working_path>)", tscl::Log::Error);
    return false;
  }


  std::filesystem::path input_path = args[1];
  std::filesystem::path working_path = args.size() == 3 ? args[2] : "runs/test";

  auto loader = std::make_shared<CITCLoader>(32, 32);
  auto &engine = loader->getPostProcessEngine();
  // engine.addTransformation(std::make_shared<image::transform::BinaryScale>());
  // engine.addTransformation(std::make_shared<image::transform::Inversion>());

  std::shared_ptr<nnet::MLPModelOptimizer<float>> optimizer =
          nnet::OptimizerFactory<float>::makeMLPModelOptimizer(
                  nnet::ModelOptimizerType::stochastic, nnet::OptimizationAlgorithm::momentum,
                  0.01f, 0.9f);

  CTParams parameters(RunPolicy::create, input_path, loader, optimizer, "runs/test");

  nnet::MLPTopology topology = {32 * 32, 64, 64, 64, 64};
  parameters.setTopology(topology);
  parameters.setMaxEpoch(100);

  CTController controller(parameters);
  logger(tscl::StringLog("Launching run", Log::Information));
  ControllerResult res = controller.run();

  if (not res) { std::cout << "ERROR: " << res << std::endl; }

  return (bool) res;
}

int main(int argc, char **argv) {
  // func_xor<float>(100, 1.0, 0.01);
  //   func_xor_batch<float>(1, 0.4, 0.00000000001);


  /*size_t sum = 0;
  for(size_t i = 0; i < 100; i++)

    std::cout << i;
    std::flush(std::cout);
    //sum += func_xor<float>(100, 0.2, 0.01);
    sum += func_xor_batch<float>(100, 0.2, 0.1);
  }*/
  // std::cout << "average : " << sum / 1000 << std::endl;*/

  Version::setCurrent(Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, VERSION_TWEAK));
  std::vector<std::string> args;
  for (size_t i = 0; i < argc; i++) args.emplace_back(argv[i]);

  return test_image(args);
  return 0;
}