#include "NeuralNetworkSerializer.hpp"

#include <filesystem>
#include <fstream>
#include <search.h>

namespace nnet {

  void NeuralNetworkSerializer::saveToFile(std::string const &path, NeuralNetworkBase const &nn,
                                           NNSerializerFlags const &flags) {
    std::ios::openmode mode = std::ios::out;
    if (flags & NNSerializerFlags::BINARY_MODE) { mode |= std::ios::binary; }

    std::ofstream file(path, mode);
    if (!file.is_open()) { throw std::runtime_error("Could not open file " + path); }
    // if the serialization fails, we want to delete the file
    try {
      saveToStream(file, nn, flags);
    } catch (std::exception &e) {
      file.close();
      std::filesystem::remove(path);
      // Re-throw the original exception
      throw;
    }
  }

  void NeuralNetworkSerializer::saveToStream(std::ostream &stream, NeuralNetworkBase const &nn,
                                             NNSerializerFlags const &flags) {
    stream << fPrecisionToStr(nn.getPrecision()) << "\n";
    if (flags & NNSerializerFlags::BINARY_MODE) {
      binarySaveToStream(stream, nn, flags);
    } else {
      AsciiSaveToStream(stream, nn, flags);
    }
  }

  void NeuralNetworkSerializer::binarySaveToStream(std::ostream &os, NeuralNetworkBase const &nn,
                                                   NNSerializerFlags const &flags) {
    auto layers = nn.getTopology();
    size_t layer_size = layers.size();
    // Write the number of layers
    os.write(reinterpret_cast<char const *>(&layer_size), sizeof(size_t));
    // output the whole array at once
    os.write(reinterpret_cast<char const *>(layers.data()), sizeof(size_t));

    auto const &activations = nn.getActivationFunctions();

    for (auto const &activation : activations) {
      std::string const &name = AFTypeToStr(activation);
      os.write(name.c_str(), static_cast<long>(name.length() * sizeof(char)));
      os.write(" ", sizeof(char));
    }

    auto f = [&]<class NN>(std::ostream &os, NN const &nn, NNSerializerFlags const &flags) {
      using real = typename NN::value_type;

      auto const &weights = nn.getWeights();
      // Write the number of weights

      for (auto const &w : weights) {
        // output the whole array at once
        size_t weight_size = w.getRows() * w.getCols();
        os.write(reinterpret_cast<char const *>(w.getData()), sizeof(real) * weight_size);
      }

      auto const &biases = nn.getBiases();
      for (auto const &w : biases) {
        // output the whole array at once
        size_t weight_size = w.getRows() * w.getCols();
        os.write(reinterpret_cast<char const *>(w.getData()), sizeof(real) * weight_size);
      }
    };

    // dynamic cast will throw if the cast failed when casting to a reference
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

  void NeuralNetworkSerializer::AsciiSaveToStream(std::ostream &os, NeuralNetworkBase const &nn,
                                                  NNSerializerFlags const &flags) {
    auto layers = nn.getTopology();
    size_t layer_size = layers.size();
    // Write the number of layers
    os << layer_size << " ";
    // output the whole array at once

    for (auto i : layers) { os << i << " "; }

    auto const &activations = nn.getActivationFunctions();

    for (auto const &activation : activations) {
      std::string const &name = af::AFTypeToStr(activation);
      os << name << " ";
    }
    os << std::endl;

    auto f = [&]<class NN>(std::ostream &os, NN const &nn, NNSerializerFlags const &flags) {
      using real = typename NN::value_type;

      auto const &weights = nn.getWeights();
      // Write the number of weights

      for (auto const &w : weights) {
        // output the whole array at once
        size_t weight_size = w.getRows() * w.getCols();
        os << w;
      }
      os << std::endl;

      auto const &biases = nn.getBiases();
      for (auto const &w : biases) {
        // output the whole array at once
        os << w;
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

  std::unique_ptr<NeuralNetworkBase> NeuralNetworkSerializer::loadFromStream(std::istream &stream) {
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

}   // namespace nnet