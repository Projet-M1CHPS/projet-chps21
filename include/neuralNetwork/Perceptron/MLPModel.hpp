#pragma once

#include "MLPSerializer.hpp"
#include "MLPerceptron.hpp"
#include "neuralNetwork/Model.hpp"

namespace nnet {
  template<typename real = float, typename = std::enable_if<std::is_floating_point_v<real>>>
  class MLPModel : public Model<real> {
  public:
    MLPModel() = default;
    ~MLPModel() = default;

    math::Matrix<real> predict(math::Matrix<real> const &input) override {
      return perceptron.predict(input);
    }

    [[nodiscard]] MLPerceptron<real> &getPerceptron() { return perceptron; }
    [[nodiscard]] MLPerceptron<real> const &getPerceptron() const { return perceptron; }

  private:
    MLPerceptron<real> perceptron;
  };

  class MLPModelMetadata {
  public:
    explicit MLPModelMetadata(MLPMetadata const &metadata) : metadata(metadata) {}

    MLPMetadata &getMLPMetadata() { return metadata; }
    MLPMetadata const &getMLPMetadata() const { return metadata; }

  private:
    MLPMetadata metadata;
  };

  template<typename real = float, typename = std::enable_if<std::is_floating_point_v<real>>>
  class MLPModelSerializer {
    using RModel = MLPModel<real>;

  public:
    MLPModelSerializer() = default;

    MLPModelMetadata const *fetchMetadata(std::filesystem::path const &p) {
      if (p == path and metadata != nullptr) { return metadata.get(); }

      MLPSerializer<real> serializer;
      tscl::logger("MLPModelSerializer: Loading metadata from " + path.string() + "...");
      auto mlp_metadata = serializer.fetchMetadataFromFile(path);

      if (not mlp_metadata) {
        tscl::logger("MLPModelSerializer: Failed to fetch metadata from file: " + path.string(),
                     tscl::Log::Error);
        return nullptr;
      }
      metadata = std::make_unique<MLPModelMetadata>(*mlp_metadata);
      return metadata.get();
    }

    MLPModelMetadata const *fetchMetadata() {
      if (metadata) { return metadata.get(); }
      return fetchMetadata(path);
    }

    std::unique_ptr<RModel> load() {
      tscl::logger("MLPModelSerializer: Loading model from " + path.string() + "...");

      if (not metadata) { fetchMetadata(); }
      if (not metadata) {
        tscl::logger("MLPModelSerializer: loading failed", tscl::Log::Error);
        return nullptr;
      }
      tscl::logger("MLPModelSerializer: loading network " + metadata->getMLPMetadata().getName());

      MLPSerializer<real> serializer;
      auto perceptron = serializer.load(path);

      auto res = std::make_unique<RModel>();
      res->getPerceptron() = std::move(*perceptron);
      return res;
    }

    std::unique_ptr<RModel> load(std::filesystem::path const &p) {
      if (p != path) {
        path = p;
        metadata = nullptr;
      }
      return load();
    }

  private:
    std::unique_ptr<MLPModelMetadata> metadata;
    std::filesystem::path path;
  };

  template<typename real = float, typename = std::enable_if<std::is_floating_point_v<real>>>
  class MLPModelFactory {
    using RModel = MLPModel<real>;

  public:
    MLPModelFactory() = delete;


    static std::unique_ptr<RModel> random(MLPTopology const &topology) {
      auto res = std::make_unique<RModel>();
      auto &mlp = res->getPerceptron();
      mlp.setTopology(topology);
      mlp.randomizeSynapses();
      return res;
    }

    static std::unique_ptr<RModel> randomSigReluAlt(MLPTopology const &topology) {
      auto res = std::make_unique<RModel>();
      auto &mlp = res->getPerceptron();
      mlp.setTopology(topology);
      mlp.randomizeWeight();
      mlp.setActivationFunction(af::ActivationFunctionType::leakyRelu);

      for (size_t i = 0; i < topology.size() - 1; i++) {
        if (i % 2 == 0 or i == topology.size() - 1) {
          mlp.setActivationFunction(af::ActivationFunctionType::sigmoid, i);
        }
      }

      return res;
    }
  };
}   // namespace nnet