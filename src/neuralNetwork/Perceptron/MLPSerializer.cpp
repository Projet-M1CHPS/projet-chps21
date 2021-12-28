#include "MLPSerializer.hpp"

namespace nnet {

  std::unique_ptr<MLPModel<float>>
  MLPModelSerializer::readFromFile(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file.is_open()) { return nullptr; }
    return readFromStream(file);
  }

  std::unique_ptr<MLPModel<float>> MLPModelSerializer::readFromStream(std::istream &stream) {
    auto res = std::make_unique<MLPModel<float>>();
    std::string line;
    std::getline(stream, line);

    // Check for header
    if (line != "#MLPModel") {
      tscl::logger("MLPModelSerializer: Invalid file format", tscl::Log::Error);
      return nullptr;
    }

    // Check for version
    // A different version may not be fatal, but we warn the user about it
    std::getline(stream, line);
    // If the version is missing from the header, we assume the file is from an version
    // And we warn the user about it, but we don't fail

    if (not line.starts_with("#Version")) {
      tscl::logger("MLPModelSerializer: version is missing from the header", tscl::Log::Warning);
      // We must rewind the stream
      stream.seekg(-static_cast<long>(line.size()), std::ios::cur);
    } else {
      tscl::Version version(line.substr(9));
      if (version.getMajor() != tscl::Version::current.getMajor() ||
          version.getMinor() != tscl::Version::current.getMinor()) {
        tscl::logger("MLPModelSerializer: loading model from a different version, compatibility is "
                     "not guaranteed, proceed with caution.",
                     tscl::Log::Debug);
        tscl::logger("MLPModelSerializer: loaded version: " + version.to_string(),
                     tscl::Log::Trace);
        tscl::logger("MLPModelSerializer: current version: " + tscl::Version::current.to_string(),
                     tscl::Log::Trace);
      }
    }

    auto perceptron = MLPerceptronSerializer::readFromStream(stream);
    res->getPerceptron() = std::move(*perceptron);

    return res;
  }

  bool MLPModelSerializer::writeToFile(const std::filesystem::path &path,
                                       const MLPModel<float> &model) {
    std::ofstream file(path);
    if (!file.is_open()) {
      tscl::logger("MLPModelSerializer: Could not open file " + path.string() + "for writing",
                   tscl::Log::Error);
      return false;
    }
    return writeToStream(file, model);
  }

  bool MLPModelSerializer::writeToStream(std::ostream &stream, const MLPModel<float> &model) {
    // Write header
    stream << "#MLPModel" << std::endl;
    stream << "#Version " << tscl::Version::current.to_string() << std::endl;

    return MLPerceptronSerializer::writeToStream(stream, model.getPerceptron());
  }

  std::unique_ptr<MLPerceptron<float>>
  MLPerceptronSerializer::readFromFile(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      tscl::logger("MLPerceptronSerializer: Could not open file " + path.string() + "for reading",
                   tscl::Log::Error);
      return nullptr;
    }
    return readFromStream(file);
  }

  std::unique_ptr<MLPerceptron<float>>
  MLPerceptronSerializer::readFromStream(std::istream &stream) {
    auto res = std::make_unique<MLPerceptron<float>>();
    std::string line;
    std::getline(stream, line);
    if (line != "#Topology") {
      tscl::logger("MLPerceptronSerializer: Invalid file format: Topology section is missing",
                   tscl::Log::Error);
      return nullptr;
    }

    // Read topology
    std::getline(stream, line);
    MLPTopology topology;
    std::stringstream ss(line);
    while (ss.good()) {
      size_t val;
      ss >> val;
      if (val == 0) {
        tscl::logger("MLPerceptronSerializer: Invalid file format: empty layer in the topology",
                     tscl::Log::Error);
        return nullptr;
      }
      topology.push_back(val);
    }
    res->setTopology(topology);
    std::getline(stream, line);

    if (line != "#ActivationFunctions") {
      tscl::logger("MLPerceptronSerializer: Invalid file format: Activation Functions section is "
                   "missing",
                   tscl::Log::Error);
      return nullptr;
    }

    // Read activation functions
    std::getline(stream, line);
    std::stringstream ss2(line);
    size_t counter = 0;
    while (ss2.good()) {
      std::string val;
      ss2 >> val;
      res->setActivationFunction(af::strToAFType(val), counter);
      counter++;
    }
    
    if (line != "#Weights") {
      tscl::logger("MLPerceptronSerializer: Invalid file format: Weights section is missing",
                   tscl::Log::Error);
      return nullptr;
    }

    auto &weights = res->getWeights();
    for (int i = 0; i < topology.size() - 1; i++) {
      float *data = weights[i].getData();
      for (int j = 0; j < weights[i].getSize(); j++) {
        float val;
        stream >> val;
        data[j] = val;
      }
    }

    stream.ignore(3);
    std::getline(stream, line);
    if (line != "#Biases") {
      tscl::logger("MLPerceptronSerializer: Invalid file format: Biases section is missing",
                   tscl::Log::Error);
      return nullptr;
    }

    auto &biases = res->getBiases();
    for (int i = 0; i < topology.size() - 1; i++) {
      float *data = biases[i].getData();
      for (int j = 0; j < biases[i].getSize(); j++) {
        float val;
        stream >> val;
        data[j] = val;
      }
    }

    return res;
  }

  bool MLPerceptronSerializer::writeToFile(const std::filesystem::path &path,
                                           const MLPerceptron<float> &perceptron) {
    std::ofstream file(path);
    if (!file.is_open()) {
      tscl::logger("MLPerceptronSerializer: Could not open file " + path.string() + "for writing",
                   tscl::Log::Error);
      return false;
    }
    return writeToStream(file, perceptron);
  }

  bool MLPerceptronSerializer::writeToStream(std::ostream &stream,
                                             const MLPerceptron<float> &perceptron) {
    // We want the maximum precision for outputing in plain text
    // This wouldn't be necessary if we were using binary files
    std::streamsize old_precision = stream.precision();
    stream << std::setprecision(std::numeric_limits<float>::max_digits10);

    // We output the topology first, so we can allocate the right amount of memory
    // when reading
    stream << "#Topology" << std::endl;
    for (size_t i = 0; i < perceptron.getTopology().size(); i++) {
      stream << perceptron.getTopology()[i];
      // No extra space at the end of the line
      if (i != perceptron.getTopology().size() - 1) { stream << " "; }
    }
    stream << std::endl;

    stream << "#ActivationFunctions" << std::endl;

    auto afs = perceptron.getActivationFunctions();
    for (const auto &af : afs) { stream << af::AFTypeToStr(af) << " "; }

    stream << "#Weights" << std::endl;
    for (auto const &weight : perceptron.getWeights()) { stream << weight << std::endl; }
    stream << "#Biases" << std::endl;
    for (auto const &bias : perceptron.getBiases()) { stream << bias << std::endl; }
    // Restore the old precision
    stream << std::setprecision(old_precision);
    return true;
  }
}   // namespace nnet