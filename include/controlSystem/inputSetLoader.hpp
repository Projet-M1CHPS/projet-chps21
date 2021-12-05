#pragma once

#include "Image.hpp"
#include "Transform.hpp"
#include "inputSet.hpp"
#include <filesystem>
#include <set>


namespace control {

  class SetLoader {
  public:
    [[nodiscard]] virtual InputSet load(std::filesystem::path const &input_path, bool verbose,
                                        std::ostream *out) = 0;

  private:
  };

  class ImageSetLoader : public SetLoader {
  public:
    ImageSetLoader() = default;

  private:
  };

  template<class TrainingSet>
  class TSLoader {
  public:
    [[nodiscard]] virtual std::shared_ptr<TrainingSet> load(std::filesystem::path const &input_path,
                                                            bool verbose, std::ostream *out) = 0;
  };
}   // namespace control
