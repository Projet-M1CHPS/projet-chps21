#pragma once

#include "image/Image.hpp"
#include "image/Transform.hpp"
#include "inputSet.hpp"
#include <filesystem>
#include <set>


namespace control {

  /** Interface for input set loaders
   *
   * This is necessary since loading may be deferred until the first time
   * the inputs are used, meaning we may not use a static method
   */
  class SetLoader {
  public:
    [[nodiscard]] virtual std::shared_ptr<InputSet> load(std::filesystem::path const &input_path,
                                                         bool verbose) = 0;
  };

  // FIXME: implement me
  class ImageSetLoader : public SetLoader {
  public:
    ImageSetLoader() = default;
  };

  /** Interface for training input set loaders
   *
   * This is necessary since loading may be deferred until the first time
   * the inputs are used, meaning we may not use a static method
   */
  class TrainingSetLoader {
  public:
    [[nodiscard]] virtual std::shared_ptr<TrainingSet>
    load(std::filesystem::path const &input_path) = 0;
  };
}   // namespace control
