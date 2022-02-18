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
        throw std::runtime_error("MLPSerializer::readTopology: #Topology section is missing");
      }

      // Read topology
      std::getline(stream, line);
      trimWhitespaces(line);

      MLPTopology res;
      std::stringstream ss(line);
      while (ss.good()) {
        size_t val;
        ss >> val;
        if (val == 0) { throw std::runtime_error("MLPSerializer::readTopology: Invalid topology"); }
        res.pushBack(val);
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
        throw std::runtime_error("MLPSerializer::readActivationFunctions: #ActivationFunctions "
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
    void readMatrices(std::istream &stream, std::vector<math::clMatrix> &matrices,
                      utils::clWrapper &wrapper, const std::string &section_name) {
      std::string line = getNextNonEmptyLine(stream);

      if (line != "#" + section_name) {
        throw std::runtime_error("MLPSerializer::readMatrices: #" + section_name +
                                 " section is "
                                 "missing");
      }

      for (auto &m : matrices) {
        math::FloatMatrix buf(m.getRows(), m.getCols());
        float *data = buf.getData();
        for (int j = 0; j < buf.getSize(); j++) {
          float val;
          stream >> val;
          data[j] = val;
        }
        m.fromFloatMatrix(buf, wrapper);
      }
    }

    void writeMatrices(std::ostream &stream, const std::vector<math::clMatrix> &matrices,
                       utils::clWrapper &wrapper, const std::string &section_name) {
      stream << "#" << section_name << std::endl;
      for (auto const &m : matrices) {
        math::FloatMatrix buf = m.toFloatMatrix(wrapper);
        stream << buf << std::endl;
      }
    }

  }   // namespace

  MLPerceptron MLPSerializer::readFromFile(utils::clWrapper &wrapper,
                                           const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("MLPerceptronSerializer: Could not open file " + path.string() +
                               "for reading");
    }
    return readFromStream(wrapper, file);
  }

  MLPerceptron MLPSerializer::readFromStream(utils::clWrapper &wrapper, std::istream &stream) {
    MLPerceptron res(wrapper);
    std::string line;
    auto topology = readTopology(stream);
    res.setTopology(topology);

    auto afs = readActivationFunctions(stream);
    if (afs.size() != topology.size() - 1) {
      throw std::runtime_error("MLPerceptronSerializer: Number of activation functions does not "
                               "match the number of layers");
    }

    for (size_t i = 0; i < afs.size(); i++) { res.setActivationFunction(afs[i], i); }

    readMatrices(stream, res.getWeights(), wrapper, "Weights");
    readMatrices(stream, res.getBiases(), wrapper, "Biases");

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

    auto cl_wrapper = perceptron.getWrapper();

    writeTopology(stream, perceptron.getTopology());
    writeActivationFunctions(stream, perceptron.getActivationFunctions());
    writeMatrices(stream, perceptron.getWeights(), cl_wrapper, "Weights");
    writeMatrices(stream, perceptron.getBiases(), cl_wrapper, "Biases");

    stream << std::setprecision(old_precision);
    return true;
  }

}   // namespace nnet