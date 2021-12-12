
#pragma once
#include "tscl.hpp"

#include "MLPerceptron.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

namespace nnet {

  class MLPMetadata {
  public:
    MLPMetadata() = default;
    MLPMetadata(std::string name, std::filesystem::path path, MLPTopology topo, tscl::Version v)
        : model_name(std::move(name)), model_path(std::move(path)), topology(std::move(topo)),
          version(std::move(v)) {}

    [[nodiscard]] std::string const &getName() const { return model_name; }
    void setName(std::string const &n) { model_name = n; }

    [[nodiscard]] std::filesystem::path const &getPath() const { return model_path; }
    [[nodiscard]] MLPTopology const &getTopology() const { return topology; }

    [[nodiscard]] tscl::Version getVersion() const { return version; }
    void setVersion(tscl::Version const &v) { version = v; }

  private:
    std::string model_name;
    std::filesystem::path model_path;
    MLPTopology topology;
    tscl::Version version;
  };

  template<typename real = float, typename = std::enable_if<std::is_floating_point_v<real>>>
  class MLPSerializer {
  public:
    MLPMetadata const *fetchMetadataFromFile(std::filesystem::path const &p) {
      if (not std::filesystem::exists(p)) { return nullptr; }
      std::ifstream file(p);
      if (not file.is_open()) { return nullptr; }

      return fetchMetadataFromStream(file, p);
    }

    MLPMetadata const *fetchMetadataFromStream(std::istream &input,
                                               std::filesystem::path const &p) {
      if (not input.good()) { return nullptr; }

      std::string line_buffer;
      std::getline(input, line_buffer);
      if (line_buffer != "MLP") {
        tscl::logger("MLPSerializer: Invalid file format.", tscl::Log::Error);
        return nullptr;
      }

      std::getline(input, line_buffer);
      if (not line_buffer.starts_with("Version")) {
        tscl::logger("MLPSerializer: Version number not found in network file", tscl::Log::Error);
        return nullptr;
      }

      tscl::Version version(line_buffer.substr(8));
      // Warn the user if the version is different from the current one
      // Not fatal, but it's not a good idea to load a network from a different version
      if (version != tscl::Version::current) {
        if (version < tscl::Version::current) {
          tscl::logger("MLPSerializer: Network file was built using an old version of the library",
                       tscl::Log::Warning);
        } else {
          tscl::logger("MLPSerializer: Network file was built using a newer version of the library",
                       tscl::Log::Warning);
        }
      }

      std::getline(input, line_buffer);
      std::string model_name;
      // The model name is optional
      if (line_buffer.starts_with("ModelName")) { model_name = line_buffer.substr(10); }

      std::getline(input, line_buffer);
      if (not line_buffer.starts_with("Topology")) {
        tscl::logger("MLPSerializer: network topology is missing", tscl::Log::Error);
        return nullptr;
      }
      MLPTopology topology = MLPTopology::fromString(line_buffer.substr(9));

      metadata = std::make_unique<MLPMetadata>(model_name, path, topology, version);
      return metadata.get();
    }

    MLPMetadata const *save(std::filesystem::path const &p) { return nullptr; }

    std::unique_ptr<MLPerceptron<real>> load();
    std::unique_ptr<MLPerceptron<real>> load(std::filesystem::path const &path);

  private:
    std::filesystem::path path;
    std::unique_ptr<MLPMetadata> metadata;
  };


}   // namespace nnet