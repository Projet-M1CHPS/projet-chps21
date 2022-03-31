#pragma once
#include "InputSetLoader.hpp"
#include "TrainingCollection.hpp"
#include "image/Image.hpp"
#include "image/Transform.hpp"
#include <filesystem>
#include <iostream>

namespace control {

  class TrainingCollectionLoader final {
  public:
    TrainingCollectionLoader(size_t tensor_size, size_t input_width, size_t input_height);

    TrainingCollection load(const std::filesystem::path &path);

    size_t getTensorSize() const { return input_set_loader.getTensorSize(); }

    size_t getInputWidth() const { return input_set_loader.getInputWidth(); }

    size_t getInputHeight() const { return input_set_loader.getInputHeight(); }

    image::transform::TransformEngine &getPreProcessEngine() {
      return input_set_loader.getPreProcessEngine();
    }

    image::transform::TransformEngine &getPostProcessEngine() {
      return input_set_loader.getPostProcessEngine();
    }


  private:
    InputSetLoader input_set_loader;
  };

}   // namespace control
