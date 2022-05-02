#pragma once
#include "InputSetLoader.hpp"
#include "TrainingCollection.hpp"
#include "image/Image.hpp"
#include "image/Transform.hpp"
#include <filesystem>
#include <iostream>

namespace control {

  /**
   * @brief Loads a training collection from a directory. The directories are expected to follow the
   * convention of training directories.
   */
  class TrainingCollectionLoader final {
  public:
    /**
     * @brief builds a loader with the specified sizes for the tensors
     * @param tensor_size The depth of the tensors to store images in
     * @param input_width The width of the image, rescaling as needed
     * @param input_height The height of the image, rescaling as needed
     */
    TrainingCollectionLoader(size_t tensor_size, size_t input_width, size_t input_height);

    /**
     * @brief Loads a training collection from a directory. The directories are expected to follow
     * the convention of training directories.
     * @param path The path to the directory.
     * @return A training collection.
     */
    TrainingCollection load(const std::filesystem::path &path);

    size_t getTensorSize() const { return input_set_loader.getTensorSize(); }

    size_t getInputWidth() const { return input_set_loader.getInputWidth(); }

    size_t getInputHeight() const { return input_set_loader.getInputHeight(); }

    /**
     * @brief The preprocessing engine to apply to the images, before rescaling.
     * @return A reference to the preprocessing engine.
     */
    image::transform::TransformEngine &getPreProcessEngine() {
      return input_set_loader.getPreProcessEngine();
    }

    /**
     * @brief The postprocessing engine to apply to the images, after rescaling.
     * @return
     */
    image::transform::TransformEngine &getPostProcessEngine() {
      return input_set_loader.getPostProcessEngine();
    }


  private:
    InputSetLoader input_set_loader;
  };

}   // namespace control
