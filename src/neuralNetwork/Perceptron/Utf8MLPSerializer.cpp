#include "Utf8MLPSerializer.hpp"

namespace nnet {

  std::string getNextNonEmptyLine(std::istream &stream) {
    std::string res;
    while (std::getline(stream, res)) {
      if (not res.empty()) { return res; }
    }
    return "";
  }

  MLPTopology Utf8MLPSerializer::readTopology(std::istream &stream) {
    std::string line = getNextNonEmptyLine(stream);
    if (line != "#Topology") {
      throw std::runtime_error("Utf8MLPSerialiazer::readTopology: #Topology section is missing");
    }

    // Read topology
    std::getline(stream, line);
    MLPTopology topology;
    std::stringstream ss(line);
    while (ss.good()) {
      size_t val;
      ss >> val;
      if (val == 0) {
        throw std::runtime_error("Utf8MLPSerialiazer::readTopology: Invalid topology");
      }
      topology.push_back(val);
    }
    return topology;
  }

  std::vector<af::ActivationFunctionType>
  Utf8MLPSerializer::readActivationFunctions(std::istream &stream) {
    std::string line = getNextNonEmptyLine(stream);

    if (line != "#ActivationFunctions") {
      throw std::runtime_error("Utf8MLPSerialiazer::readActivationFunctions: #ActivationFunctions "
                               "section is missing");
    }

    // Read activation functions
    std::getline(stream, line);
    std::stringstream ss2(line);
    std::vector<af::ActivationFunctionType> res;
    while (ss2.good()) {
      std::string val;
      ss2 >> val;
      res.push_back(af::strToAFType(val));
    }
    return res;
  }

  std::vector<math::FloatMatrix> Utf8MLPSerializer::readMatrices(std::istream &stream,
                                                                 const MLPTopology &topology,
                                                                 const std::string &section_name) {
    std::string line = getNextNonEmptyLine(stream);

    if (line != "#" + section_name) {
      throw std::runtime_error("Utf8MLPSerialiazer::readMatrices: #" + section_name +
                               " section is "
                               "missing");
    }

    std::vector<math::FloatMatrix> res;
    res.reserve(topology.size());
    for (size_t i = 0; i < topology.size(); i++) { res.emplace_back(topology[i], topology[i + 1]); }

    for (int i = 0; i < topology.size() - 1; i++) {
      float *data = res[i].getData();
      for (int j = 0; j < res[i].getSize(); j++) {
        float val;
        stream >> val;
        data[j] = val;
      }
    }
    return res;
  }


  MLPerceptron<float> Utf8MLPSerializer::readFromFile(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("MLPerceptronSerializer: Could not open file " + path.string() +
                               "for reading");
    }
    return readFromStream(file);
  }

  MLPerceptron<float> Utf8MLPSerializer::readFromStream(std::istream &stream) {
    MLPerceptron<float> res;
    std::string line;
    auto topology = readTopology(stream);
    res.setTopology(topology);

    auto afs = readActivationFunctions(stream);
    if (afs.size() != topology.size() - 1) {
      throw std::runtime_error("MLPerceptronSerializer: Number of activation functions does not "
                               "match the number of layers");
    }

    for (size_t i = 0; i < afs.size(); i++) { res.setActivationFunction(afs[i], i); }

    auto weights = readMatrices(stream, topology, "Weights");
    auto biases = readMatrices(stream, topology, "Biases");

    res.getWeights() = std::move(weights);
    res.getBiases() = std::move(biases);

    return res;
  }

  void Utf8MLPSerializer::writeTopology(std::ostream &stream, const MLPTopology &topology) {
    // We output the topology first, so we can allocate the right amount of memory
    // when reading
    stream << "#Topology" << std::endl;
    for (unsigned long i : topology) { stream << i << " "; }
    stream << std::endl;
  }

  void Utf8MLPSerializer::writeActivationFunctions(
          std::ostream &stream, const std::vector<af::ActivationFunctionType> &functions) {
    stream << "#ActivationFunctions" << std::endl;
    for (const auto &af : functions) { stream << af::AFTypeToStr(af) << " "; }
  }

  void Utf8MLPSerializer::writeMatrices(std::ostream &stream,
                                        const std::vector<math::FloatMatrix> &matrices,
                                        const std::string &section_name) {
    stream << "#" << section_name << std::endl;
    for (auto const &m : matrices) { stream << m << std::endl; }
  }


  bool Utf8MLPSerializer::writeToFile(const std::filesystem::path &path,
                                      const MLPerceptron<float> &perceptron) {
    std::ofstream file(path);
    if (!file.is_open()) {
      tscl::logger("MLPerceptronSerializer: Could not open file " + path.string() + "for writing",
                   tscl::Log::Error);
      return false;
    }
    return writeToStream(file, perceptron);
  }

  bool Utf8MLPSerializer::writeToStream(std::ostream &stream,
                                        const MLPerceptron<float> &perceptron) {
    // We want the maximum precision for outputing in plain text
    // This wouldn't be necessary if we were using binary files
    std::streamsize old_precision = stream.precision();
    stream << std::setprecision(std::numeric_limits<float>::max_digits10);

    writeTopology(stream, perceptron.getTopology());
    writeActivationFunctions(stream, perceptron.getActivationFunctions());
    writeMatrices(stream, perceptron.getWeights(), "Weights");
    writeMatrices(stream, perceptron.getBiases(), "Biases");

    stream << std::setprecision(old_precision);
    return true;
  }

}   // namespace nnet