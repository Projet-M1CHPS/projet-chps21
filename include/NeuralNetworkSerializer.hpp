#pragma once
#include "NeuralNetwork.hpp"
#include "Utils.hpp"

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
    /**
     * @brief If set, the serializer will not load/write the weights to/from the
     * output file
     *
     */
    NO_WEIGHTS = 1,

    /**
     * @brief If set, the serializer will not load/write the biases to/from the
     * output file
     *
     */
    NO_BIASES = 2,

    /**
     * @brief If set, the serializer will not load/write the activations
     * function to/from the file
     *
     */
    NO_ACTIVATIONS = 4,

    /**
     * @brief If set, the serializer will output in binary mode
     *
     */
    BINARY_MODE = 16,
  };

  // This is an helper class without need for an instance
  NeuralNetworkSerializer() = delete;
  NeuralNetworkSerializer(const NeuralNetworkSerializer &) = delete;
  NeuralNetworkSerializer &operator=(const NeuralNetworkSerializer &) = delete;

  static void saveToFile(std::string const &path, NeuralNetworkBase const &nn,
                         NNSerializerFlags const &flags);

  static void saveToStream(std::ostream &stream, NeuralNetworkBase const &nn,
                           NNSerializerFlags const &flags);

  static std::unique_ptr<NeuralNetworkBase>
  loadFromFile(std::string const &path);

  static std::unique_ptr<NeuralNetworkBase>
  loadFromStream(std::istream &stream);

  static void saveMetadataToJSon(std::string const &path,
                                 NeuralNetworkBase const &nn);

  static std::unique_ptr<NeuralNetworkBase>
  loadMetadataFromJSon(std::string const &path);

private:
  static void binarySaveToStream(std::ostream& os, NeuralNetworkBase const &nn,
                                 NNSerializerFlags const &flags);
  static void saveToStream(std::ostream& os, NeuralNetworkBase const &nn,
                                 NNSerializerFlags const &flags);

  static void binaryLoadFromStream();
  static void loadFromStream();
};

} // namespace nnet