#include "NeuralNetworkSerializer.hpp"

#include <fstream>

namespace nnet {
void NeuralNetworkSerializer::saveToFile(std::string const &path,
                                         NeuralNetworkBase const &nn,
                                         NNSerializerFlags const &flags) {

  std::ios::openmode mode = std::ios::out;
  if (flags & NNSerializerFlags::BINARY_MODE)
    mode |= std::ios::binary;

  std::ofstream file(path, mode);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file " + path);
  }
  saveToStream(file, nn, flags);
}

void NeuralNetworkSerializer::saveToStream(std::ostream &stream,
                                           NeuralNetworkBase const &nn,
                                           NNSerializerFlags const &flags) {

  // The first byte is the format
  if (flags & NNSerializerFlags::BINARY_MODE)
    stream << "b ";
  else
    stream << "a ";
  stream << fPrecisionToStr(nn.getPrecision()) << "\n";
}

void NeuralNetworkSerializer::binarySaveToStream(
    std::ostream &os, NeuralNetworkBase const &nn,
    NNSerializerFlags const &flags) {

  auto layers = nn.getLayersSize();
  size_t layer_size = layers.size();
  // Write the number of layers
  os.write(reinterpret_cast<char const *>(&layer_size), sizeof(size_t));
  // output the whole array at once
  os.write(reinterpret_cast<char const *>(layers.data()), sizeof(size_t));

  if (not(flags & NNSerializerFlags::NO_ACTIVATIONS)) {
    auto const &activations = nn.getActivationFunctions();

    for (auto const &activation : activations) {
      std::string const &name = AFTypeToStr(activation);
      os.write(name.c_str(), name.size() * sizeof(char));
      os.write("\0", sizeof(char));
    }
  }

  auto f = [&]<class NN>(std::ostream &os, NN const &nn,
                         NNSerializerFlags const &flags) {
    typedef typename NN::value_type real;

    if (not(flags & NNSerializerFlags::NO_WEIGHTS)) {
      auto const &weights = nn.getWeights();
      // Write the number of weights

      for (auto const &w : weights) {
        // output the whole array at once
        size_t weight_size = w.getRows() * w.getCols();
        os.write(reinterpret_cast<char const *>(w.getData()),
                 sizeof(real) * weight_size);
      }
    }

    if (not(flags & NNSerializerFlags::NO_BIASES)) {
      auto const &biases = nn.getBiases();
      for (auto const &w : biases) {
        // output the whole array at once
        size_t weight_size = w.getRows() * w.getCols();
        os.write(reinterpret_cast<char const *>(w.getData()),
                 sizeof(real) * weight_size);
      }
    }
  };

  if (nn.getPrecision() == FloatingPrecision::float32) {
    auto const &nn_ = dynamic_cast<NeuralNetwork<float> const &>(nn);
    f(os, nn_, flags);
  } else if (nn.getPrecision() == FloatingPrecision::float64) {
    auto const &nn_ = dynamic_cast<NeuralNetwork<double> const &>(nn);
    f(os, nn_, flags);
  } else {
    throw std::runtime_error("Unsupported precision");
  }
}

void NeuralNetworkSerializer::saveToStream(std::ostream &os,
                                           NeuralNetworkBase const &nn,
                                           NNSerializerFlags const &flags) {
  auto layers = nn.getLayersSize();
  size_t layer_size = layers.size();
  // Write the number of layers
  os << layer_size << " ";
  // output the whole array at once
  std::copy(layers.begin(), layers.end(), std::ostream_iterator<int>(os, " "));

  if (not(flags & NNSerializerFlags::NO_ACTIVATIONS)) {
    auto const &activations = nn.getActivationFunctions();

    for (auto const &activation : activations) {
      std::string const &name = AFTypeToStr(activation);
      os << name << " ";
    }
  }

  auto f = [&]<class NN>(std::ostream &os, NN const &nn,
                         NNSerializerFlags const &flags) {
    typedef typename NN::value_type real;

    if (not(flags & NNSerializerFlags::NO_WEIGHTS)) {
      auto const &weights = nn.getWeights();
      // Write the number of weights

      for (auto const &w : weights) {
        // output the whole array at once
        size_t weight_size = w.getRows() * w.getCols();
        os.write(reinterpret_cast<char const *>(w.getData()),
                 sizeof(real) * weight_size);
      }
    }

    if (not(flags & NNSerializerFlags::NO_BIASES)) {
      auto const &biases = nn.getBiases();
      for (auto const &w : biases) {
        // output the whole array at once
        size_t weight_size = w.getRows() * w.getCols();
        os.write(reinterpret_cast<char const *>(w.getData()),
                 sizeof(real) * weight_size);
      }
    }
  };

  if (nn.getPrecision() == FloatingPrecision::float32) {
    auto const &nn_ = dynamic_cast<NeuralNetwork<float> const &>(nn);
    f(os, nn_, flags);
  } else if (nn.getPrecision() == FloatingPrecision::float64) {
    auto const &nn_ = dynamic_cast<NeuralNetwork<double> const &>(nn);
    f(os, nn_, flags);
  } else {
    throw std::runtime_error("Unsupported precision");
  }
}

std::unique_ptr<NeuralNetworkBase>
NeuralNetworkSerializer::loadFromFile(std::string const &path) {
  utils::error("FIXME: loadFromFile Not implemented");
}

std::unique_ptr<NeuralNetworkBase>
NeuralNetworkSerializer::loadFromStream(std::istream &stream) {
  utils::error("FIXME: loadFromStream Not implemented");
}

void NeuralNetworkSerializer::saveMetadataToJSon(std::string const &path,
                                                 NeuralNetworkBase const &nn) {
  utils::error("FIXME: SaveMetadataToJSon Not implemented");
}

std::unique_ptr<NeuralNetworkBase>
NeuralNetworkSerializer::loadMetadataFromJSon(std::string const &path) {
  utils::error("FIXME: LoadMetadataFromJSon Not implemented");
}

} // namespace nnet