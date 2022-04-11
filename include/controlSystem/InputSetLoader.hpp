#pragma once
#include "InputSet.hpp"
#include "math/clFMatrix.hpp"
#include "math/clFTensor.hpp"
#include "image/Image.hpp"
#include "image/Transform.hpp"
#include <iostream>

namespace control {

  class InputSetLoader {
  public:
    InputSetLoader(size_t tensor_size, size_t input_width, size_t input_height)
        : tensor_size(tensor_size), input_width(input_width), input_height(input_height) {}


    InputSet load(const std::filesystem::path &path, bool load_classes, bool shuffle_samples) const;

    size_t getTensorSize() const { return tensor_size; }
    size_t getInputWidth() const { return input_width; }
    size_t getInputHeight() const { return input_height; }

    image::transform::TransformEngine &getPreProcessEngine() { return preprocess_engine; }
    image::transform::TransformEngine &getPostProcessEngine() { return postprocess_engine; }

  private:

    InputSet loadWithClasses(const std::filesystem::path &path, bool shuffle_samples) const;
    InputSet loadWithoutClasses(const std::filesystem::path &path, bool shuffle_samples) const;

    image::transform::TransformEngine preprocess_engine, postprocess_engine;
    size_t tensor_size, input_width, input_height;
  };

};   // namespace control
