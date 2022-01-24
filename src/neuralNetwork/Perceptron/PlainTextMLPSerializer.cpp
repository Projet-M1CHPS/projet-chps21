#include "PlainTextMLPSerializer.hpp"

namespace nnet {

  std::string getNextNonEmptyLine(std::istream &stream) {
    std::string res;
    while (std::getline(stream, res)) {
      if (not res.empty() and res.find_first_not_of(" \t\n\r\f\v") != std::string::npos) {
        return res;
      }
    }
    return "";
  }

  void trimWhitespaces(std::string &str) {
    str.erase(0, str.find_first_not_of(" \t\n\r\f\v"));
    str.erase(str.find_last_not_of(" \t\n\r\f\v") + 1);
  }

  MLPTopology PlainTextMLPSerializer::readTopology(std::istream &stream) {
    std::string line = getNextNonEmptyLine(stream);
    if (line != "#Topology") {
      throw std::runtime_error("Utf8MLPSerialiazer::readTopology: #Topology section is missing");
    }

    // Read topology
    std::getline(stream, line);
    trimWhitespaces(line);
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
  PlainTextMLPSerializer::readActivationFunctions(std::istream &stream) {
    std::string line = getNextNonEmptyLine(stream);

    if (line != "#ActivationFunctions") {
      throw std::runtime_error("Utf8MLPSerialiazer::readActivationFunctions: #ActivationFunctions "
                               "section is missing");
    }

    // Read activation functions
    std::getline(stream, line);
    trimWhitespaces(line);
    std::stringstream ss2(line);
    std::vector<af::ActivationFunctionType> res;
    while (ss2.good()) {
      std::string val;
      ss2 >> val;
      res.push_back(af::strToAFType(val));
    }
    return res;
  }

  void PlainTextMLPSerializer::readMatrices(std::istream &stream,
                                            std::vector<math::FloatMatrix> &matrices,
                                            const std::string &section_name) {
    std::string line = getNextNonEmptyLine(stream);

    if (line != "#" + section_name) {
      throw std::runtime_error("Utf8MLPSerialiazer::readMatrices: #" + section_name +
                               " section is "
                               "missing");
    }

    for (auto &m : matrices) {
      float *data = m.getData();
      for (int j = 0; j < m.getSize(); j++) {
        float val;
        stream >> val;
        data[j] = val;
      }
    }
  }


  MLPerceptron PlainTextMLPSerializer::readFromFile(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("MLPerceptronSerializer: Could not open file " + path.string() +
                               "for reading");
    }
    return readFromStream(file);
  }

  MLPerceptron PlainTextMLPSerializer::readFromStream(std::istream &stream) {
    MLPerceptron res;
    std::string line;
    auto topology = readTopology(stream);
    res.setTopology(topology);

    auto afs = readActivationFunctions(stream);
    if (afs.size() != topology.size() - 1) {
      throw std::runtime_error("MLPerceptronSerializer: Number of activation functions does not "
                               "match the number of layers");
    }

    for (size_t i = 0; i < afs.size(); i++) { res.setActivationFunction(afs[i], i); }

    readMatrices(stream, res.getWeights(), "Weights");
    readMatrices(stream, res.getBiases(), "Biases");

    return res;
  }

  void PlainTextMLPSerializer::writeTopology(std::ostream &stream, const MLPTopology &topology) {
    // We output the topology first, so we can allocate the right amount of memory
    // when reading
    stream << "#Topology" << std::endl;
    for (unsigned long i : topology) { stream << i << " "; }
    stream << std::endl;
  }

  void PlainTextMLPSerializer::writeActivationFunctions(
          std::ostream &stream, const std::vector<af::ActivationFunctionType> &functions) {
    stream << "#ActivationFunctions" << std::endl;
    for (const auto &af : functions) { stream << af::AFTypeToStr(af) << " "; }
    stream << std::endl;
  }

  void PlainTextMLPSerializer::writeMatrices(std::ostream &stream,
                                             const std::vector<math::FloatMatrix> &matrices,
                                             const std::string &section_name) {
    stream << "#" << section_name << std::endl;
    for (auto const &m : matrices) { stream << m << std::endl; }
  }


  bool PlainTextMLPSerializer::writeToFile(const std::filesystem::path &path,
                                           const MLPerceptron &perceptron) {
    std::ofstream file(path);
    if (!file.is_open()) {
      tscl::logger("MLPerceptronSerializer: Could not open file " + path.string() + "for writing",
                   tscl::Log::Error);
      return false;
    }
    return writeToStream(file, perceptron);
  }

  bool PlainTextMLPSerializer::writeToStream(std::ostream &stream,
                                             const MLPerceptron &perceptron) {
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