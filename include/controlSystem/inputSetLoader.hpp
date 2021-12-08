#pragma once

#include "image/Image.hpp"
#include "image/Transform.hpp"
#include "inputSet.hpp"
#include <filesystem>
#include <set>


namespace control {

  /** Base class for loading an input set
   *
   * @tparam real Precision of the input set
   */
  template<typename real, typename = std::enable_if<std::is_floating_point_v<real>>>
  class SetLoader {
  public:
    [[nodiscard]] virtual InputSet<real> load(std::filesystem::path const &input_path, bool verbose,
                                              std::ostream *out) = 0;

  private:
  };

  // FIXME: implement me
  template<typename real>
  class ImageSetLoader : public SetLoader<real> {
  public:
    ImageSetLoader() = default;

  private:
  };

  /** Base class for training set loaders
   *
   * @tparam TrainingSet The type of training set to load
   */
  template<class TrainingSet>
  class TSLoader {
  public:
    [[nodiscard]] virtual std::shared_ptr<TrainingSet> load(std::filesystem::path const &input_path,
                                                            bool verbose, std::ostream *out) = 0;
  };
}   // namespace control
