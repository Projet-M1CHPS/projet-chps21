#include <nlohmann/json.hpp>

#include "Control.hpp"
#include "Network.hpp"
#include "ProjectVersion.hpp"
#include "tscl.hpp"


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

  [[nodiscard]] json readJSONConfigFile() const;

  [[nodiscard]] static bool checkConfigValid(const json &config);

  static void setupLogger();

public:
  explicit NetworkInterface(std::string parameter_filepath)
      : parameterFilepath(std::move(parameter_filepath)){};
  ~NetworkInterface() = default;

  [[nodiscard]] json getJSONConfig() const { return readJSONConfigFile(); }
  void printJSONConfig() const { std::cout << std::setw(4) << getJSONConfig() << std::endl; }

  bool createAndTrain();
};