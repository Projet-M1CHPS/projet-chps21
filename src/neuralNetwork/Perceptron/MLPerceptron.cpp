#include "MLPerceptron.hpp"

namespace nnet {

  namespace {
    void applyAF(af::ActivationFunctionType type, math::clFMatrix& mat, utils::clWrapper& wrapper, cl::CommandQueue& queue) {

      if (type == af::ActivationFunctionType::identity)
        return;
      auto afunc = af::getAFKernelFromType(type, wrapper).first;
      afunc.setArg(0, mat.getBuffer());
      queue.enqueueNDRangeKernel(afunc, cl::NullRange, mat.size(), cl::NullRange);
    }
  }

  MLPTopology MLPTopology::fromString(const std::string &str) {
    std::vector<size_t> layers;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, ',')) { layers.push_back(std::stoi(token)); }
    return MLPTopology(layers);
  }

  MLPerceptron::MLPerceptron(utils::clWrapper *wrapper, const MLPTopology &topology) {
    if (not topology.empty()) { setTopology(topology); }
    this->wrapper = wrapper;
  }

  math::clFMatrix MLPerceptron::predict(math::clFMatrix const &input) const {
    const size_t nbInput = input.getRows();

    if (nbInput != weights.front().getCols()) {
      throw std::invalid_argument("Invalid number of input");
    }

    cl::CommandQueue queue(wrapper->getDefaultDevice());

    auto current_layer = math::clFMatrix::gemm(1.0f, false, weights[0], false, input, 1.0f,
                                               biases[0], *wrapper, queue);

    applyAF(activation_functions[0], current_layer, *wrapper, queue);

    for (size_t i = 1; i < weights.size(); i++) {
      current_layer = math::clFMatrix::gemm(1.0f, false, weights[i], false, current_layer, 1.0f,
                                            biases[i], *wrapper, queue);

      applyAF(activation_functions[i], current_layer, *wrapper, queue);
    }
    queue.finish();
    return current_layer;
  }


  void MLPerceptron::setTopology(MLPTopology const &topology) {
    if (topology.empty()) return;
    if (topology.size() < 2) { throw std::invalid_argument("Requires atleast 2 layers"); }

    weights.clear();
    biases.clear();
    for (size_t i = 0; i < topology.size() - 1; i++) {
      // Create a matrix of size (layers[i + 1] x layers[i])
      // So that each weight matrix can be multiplied by the previous layer
      weights.emplace_back(topology[i + 1], topology[i], *wrapper);
      biases.emplace_back(topology[i + 1], 1, *wrapper);
      activation_functions.push_back(af::ActivationFunctionType::sigmoid);
    }
    this->topology = topology;
  }

  void MLPerceptron::randomizeWeight() {
    for (auto &layer : weights) {
      float x = std::sqrt(2.0f / (float) layer.getRows());
      math::FloatMatrix buf(layer.getRows(), layer.getCols());
      math::randomize<float>(buf, -x, x);
      layer.fromFloatMatrix(buf, *wrapper);
    }

    for (auto &layer : biases) {
      float x = std::sqrt(2.0f / (float) layer.getRows());
      math::FloatMatrix buf(layer.getRows(), layer.getCols());
      math::randomize<float>(buf, -x, x);
      layer.fromFloatMatrix(buf, *wrapper);
    }
  }

  std::ostream &operator<<(std::ostream &os, const MLPerceptron &nn) {
    auto wrapper = nn.getWrapper();
    const size_t size = nn.getWeights().size();
    os << "-------input-------\n";
    for (size_t i = 0; i < size; i++) {
      os << "-----weight[" << i << "]-----\n";
      // We need to fetch the matrix from the defice to print it
      auto buf1 = nn.getWeights()[i].toFloatMatrix(wrapper);
      os << buf1;
      os << "------bias[" << i << "]------\n";
      auto buf2 = nn.getBiases()[i].toFloatMatrix(wrapper);
      os << buf2;
      if (i != size - 1) { os << "-----hidden[" << i << "]-----\n"; }
    }
    os << "-------output------\n";
    return os;
  }

}   // namespace nnet