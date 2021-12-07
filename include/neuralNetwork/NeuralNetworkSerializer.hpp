#pragma once

#include "NeuralNetwork.hpp"
#include "Utils.hpp"
#include <iterator>

namespace nnet {

  /**
   * @brief Helper class for serializing and deserializing neural networks
   * Also offers utility for outputing a network metadata as a JSon file
   *
   * @todo Implement me !
   *
   */
  class NeuralNetworkSerializer {
  public:
    enum NNSerializerFlags {
      DEFAULT = 0,
      BINARY_MODE = 1,
    };

    // This is an helper class without need for an instance
    NeuralNetworkSerializer() = delete;
    NeuralNetworkSerializer(const NeuralNetworkSerializer &) = delete;
    NeuralNetworkSerializer &operator=(const NeuralNetworkSerializer &) = delete;

    static void saveToFile(std::string const &path, NeuralNetworkBase const &nn,
                           NNSerializerFlags const &flags = DEFAULT);

    static void saveToStream(std::ostream &stream, NeuralNetworkBase const &nn,
                             NNSerializerFlags const &flags = DEFAULT);

    static std::unique_ptr<NeuralNetworkBase>
    loadFromFile(std::string const &path);

    static std::unique_ptr<NeuralNetworkBase>
    loadFromStream(std::istream &stream);

    static void saveMetadataToJSon(std::string const &path,
                                   NeuralNetworkBase const &nn);

    static std::unique_ptr<NeuralNetworkBase>
    loadMetadataFromJSon(std::string const &path);

  private:
    static void binarySaveToStream(std::ostream &os, NeuralNetworkBase const &nn,
                                   NNSerializerFlags const &flags = DEFAULT);
    static void AsciiSaveToStream(std::ostream &os, NeuralNetworkBase const &nn,
                                  NNSerializerFlags const &flags = DEFAULT);

    static void binaryLoadFromStream();
    static void AsciiLoadFromStream();
  };

}   // namespace nnet