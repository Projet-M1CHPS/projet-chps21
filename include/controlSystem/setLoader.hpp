#pragma once

#include "Image.hpp"
#include "Transform.hpp"
#include "inputSet.hpp"
#include <filesystem>


namespace control {

  class SetLoader {
  public:
    [[nodiscard]] virtual InputSet load(std::filesystem::path const &input_path, bool verbose,
                                        std::ostream *out) = 0;

  private:
  };

  class ImageSetLoader : public SetLoader {
  public:
  private:
  };

  class TrainingSetLoader {
  public:
    [[nodiscard]] virtual TrainingSet load(std::filesystem::path const &input_path, bool verbose,
                                           std::ostream *out) = 0;
  };

  class ImageTrainingSetLoader : public TrainingSetLoader {
  public:
    ImageTrainingSetLoader(size_t width, size_t height)
        : target_width(width), target_height(height) {}

    [[nodiscard]] TrainingSet load(std::filesystem::path const &input_path, bool verbose,
                                   std::ostream *out) override;

    [[nodiscard]] image::transform::TransformEngine &getPreProcessEngine() { return pre_process; }
    [[nodiscard]] image::transform::TransformEngine const &getPreProcessEngine() const {
      return pre_process;
    }

    [[nodiscard]] image::transform::TransformEngine &getPostProcessEngine() { return post_process; }
    [[nodiscard]] image::transform::TransformEngine const &getPostProcessEngine() const {
      return post_process;
    }

  private:
    image::transform::TransformEngine pre_process;
    image::transform::TransformEngine post_process;
    size_t target_width, target_height;
  };
}   // namespace control
