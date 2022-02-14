#include "MLPSerializer.hpp"
#include <fstream>
#include <iostream>
#include <tscl.hpp>

namespace nnet {

  namespace {
    // Parse the stream until a non-empty line is found
    // Or EOF is reached
    std::string getNextNonEmptyLine(std::istream &stream) {
      std::string res;
      while (std::getline(stream, res)) {
        if (not res.empty() and res.find_first_not_of(" \t\n\r\f\v") != std::string::npos) {
          return res;
        }
      }
      return "";
    }

    // Removes beginning and ending spaces and tabs
    void trimWhitespaces(std::string &str) {
      str.erase(0, str.find_first_not_of(" \t\n\r\f\v"));
      str.erase(str.find_last_not_of(" \t\n\r\f\v") + 1);
    }

    // Reads a topology from a stream
    MLPTopology readTopology(std::istream &stream) {
      std::string line = getNextNonEmptyLine(stream);
      if (line != "#Topology") {
        throw std::runtime_error("MLPSerialiazer::readTopology: #Topology section is missing");
      }

      // Read topology
      std::getline(stream, line);
      trimWhitespaces(line);

      MLPTopology res;
      std::stringstream ss(line);
      while (ss.good()) {
        size_t val;
        ss >> val;
        if (val == 0) {
          throw std::runtime_error("MLPSerialiazer::readTopology: Invalid topology");
        }
        res.push_back(val);
      }
      return res;
    }

    void writeTopology(std::ostream &stream, const MLPTopology &topology) {
      // We output the topology first, so we can allocate the right amount of memory
      // when reading
      stream << "#Topology" << std::endl;
      // Write each layer size, separated by a whitespace
      for (unsigned long i : topology) { stream << i << " "; }
      stream << std::endl;
    }

    // Reads the activation functions from a file
    std::vector<af::ActivationFunctionType> readActivationFunctions(std::istream &stream) {
      std::string line = getNextNonEmptyLine(stream);

      if (line != "#ActivationFunctions") {
        throw std::runtime_error("MLPSerialiazer::readActivationFunctions: #ActivationFunctions "
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

    void writeActivationFunctions(std::ostream &stream,
                                  const std::vector<af::ActivationFunctionType> &functions) {
      stream << "#ActivationFunctions" << std::endl;
      for (const auto &af : functions) { stream << af::AFTypeToStr(af) << " "; }
      stream << std::endl;
    }

    // Fill the matrices with the values from the stream
    // Matrices should already be allocated, and the size should match the topology
    void readMatrices(std::istream &stream, std::vector<math::FloatMatrix> &matrices,
                      const std::string &section_name) {
      std::string line = getNextNonEmptyLine(stream);

      if (line != "#" + section_name) {
        throw std::runtime_error("MLPSerialiazer::readMatrices: #" + section_name +
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

    void writeMatrices(std::ostream &stream, const std::vector<math::FloatMatrix> &matrices,
                       const std::string &section_name) {
      stream << "#" << section_name << std::endl;
      for (auto const &m : matrices) { stream << m << std::endl; }
    }

  }   // namespace

  MLPerceptron MLPSerializer::readFromFile(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("MLPerceptronSerializer: Could not open file " + path.string() +
                               "for reading");
    }
    return readFromStream(file);
  }

  MLPerceptron MLPSerializer::readFromStream(std::istream &stream) {
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

  bool MLPSerializer::writeToFile(const std::filesystem::path &path,
                                  const MLPerceptron &perceptron) {
    std::ofstream file(path);
    if (!file.is_open()) {
      tscl::logger("MLPerceptronSerializer: Could not open file " + path.string() + "for writing",
                   tscl::Log::Error);
      return false;
    }
    return writeToStream(file, perceptron);
  }

  bool MLPSerializer::writeToStream(std::ostream &stream, const MLPerceptron &perceptron) {
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