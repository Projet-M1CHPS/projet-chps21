#pragma once
#include "Image.hpp"
#include "NeuralNetwork.hpp"
#include "NeuralNetworkSerializer.hpp"
#include "Transform.hpp"
#include <filesystem>
#include <memory>
#include <vector>

class ImageCache {
public:
  enum Flags {
    deferred = 0,
    parallel = 1,
  };

private:
};

inline size_t operator"" _gb(unsigned long long x) { return x * 1000000000; }

inline size_t operator"" _mb(unsigned long long x) { return x * 1000000; }

inline size_t operator"" _kb(unsigned long long x) { return x * 1000; }

inline size_t operator"" _byte(unsigned long long x) { return x; }

class RunConfiguration {
public:
  enum Flags : unsigned { reuse_network = 1, save_network = 2, reuse_cache = 4, keep_cache = 8 };

  enum Mode : unsigned { runMode = 0, trainMode };

  RunConfiguration();
  RunConfiguration(std::filesystem::path input_path, std::filesystem::path working_dir);

  bool operator==(RunConfiguration const &other) const;

  [[nodiscard]] std::filesystem::path const &getWorkingDirectory() const;
  [[nodiscard]] std::filesystem::path const &getInputPath() const;

  [[nodiscard]] unsigned getCacheFlags() const;
  [[nodiscard]] size_t getCacheSize() const;

  [[nodiscard]] std::vector<image::transform::TransformType> const &getTransformations() const;

  [[nodiscard]] std::vector<af::ActivationFunctionType> const &getActivationFunctions() const;
  [[nodiscard]] nnet::FloatingPrecision getFPPrecision() const;

  [[nodiscard]] std::vector<size_t> const &getTopology() const;

  [[nodiscard]] unsigned getMode() const { return mode; }

private:
  std::filesystem::path input_path, working_dir;
  unsigned flags;
  Mode mode;

  unsigned cache_flags;
  size_t cache_size;

  std::vector<image::transform::TransformType> transformations;

  std::vector<size_t> topology;
  std::vector<af::ActivationFunctionType> activation_functions;
  nnet::FloatingPrecision precision;
};

class WorkingEnvironnement {
public:
  WorkingEnvironnement() = default;
  static WorkingEnvironnement make(std::filesystem::path working_dir);

  RunConfiguration loadConfiguration() const;
  void cleanup(RunConfiguration const &config) const;

  std::filesystem::path getCachePath() const;
  std::filesystem::path getNeuralNetworkPath() const;

private:
  explicit WorkingEnvironnement(std::filesystem::path working_dir);

  std::filesystem::path working_dir;
};

bool runOnConfig(RunConfiguration const &config);