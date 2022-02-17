#include <nlohmann/json.hpp>

#include "Control.hpp"
#include "Network.hpp"
#include "ProjectVersion.hpp"
#include "tscl.hpp"


#include <csignal>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

using namespace control;
using namespace control::classifier;
using namespace tscl;
using json = nlohmann::json;


class NetworkInterface {
private:
  std::string parameterFilepath;

  static void initSignalHandler();

  /// \return Raw JSON from the config file
  [[nodiscard]] static json readJSONConfig(const std::string &config_file_path);
  [[nodiscard]] json readJSONConfig() const;

  /// Will throw if an error is detected
  static void checkConfigValid(const json &config);

  static void setupLogger();

public:
  explicit NetworkInterface(std::string parameter_filepath)
      : parameterFilepath(std::move(parameter_filepath)){};
  explicit NetworkInterface() = default;
  ;
  ~NetworkInterface() = default;

  ///
  /// \return readJSONConfig() with checked & filtered attributes.
  [[nodiscard]] json getJSONConfig() const;

  void printJSONConfig() const;

  void onPrecisionChanged(const std::function<void(float)> &callback) const;

  bool createAndTrain();
};