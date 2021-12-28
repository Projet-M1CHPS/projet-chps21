#include "Utf8MLPModelSerializer.hpp"
#include "Utf8MLPSerializer.hpp"
#include <fstream>

namespace nnet {

  MLPModel<float> Utf8MLPModelSerializer::readFromFile(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file.is_open()) { throw std::runtime_error("Could not open file: " + path.string()); }
    return readFromStream(file);
  }

  MLPModel<float> Utf8MLPModelSerializer::readFromStream(std::istream &stream) {
    MLPModel<float> res;
    std::string line;
    std::getline(stream, line);

    // Check for header
    if (line != "#MLPModel") {
      throw std::runtime_error(
              "Utf8MLPModelSerializer::readFromStream: File is not a valid MLPModel file");
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

    Utf8MLPSerializer serializer;
    auto perceptron = serializer.readFromStream(stream);
    res.getPerceptron() = std::move(perceptron);

    return res;
  }

  bool Utf8MLPModelSerializer::writeToFile(const std::filesystem::path &path,
                                           const MLPModel<float> &model) {
    std::ofstream file(path);
    if (!file.is_open()) {
      tscl::logger("MLPModelSerializer: Could not open file " + path.string() + "for writing",
                   tscl::Log::Error);
      return false;
    }
    return writeToStream(file, model);
  }

  bool Utf8MLPModelSerializer::writeToStream(std::ostream &stream, const MLPModel<float> &model) {
    // Write header
    stream << "#MLPModel" << std::endl;
    stream << "#Version " << tscl::Version::current.to_string() << std::endl;

    Utf8MLPSerializer serializer;

    return serializer.writeToStream(stream, model.getPerceptron());
  }
}   // namespace nnet