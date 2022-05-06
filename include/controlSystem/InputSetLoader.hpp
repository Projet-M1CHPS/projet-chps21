#pragma once
#include "InputSet.hpp"
#include "image/Image.hpp"
#include "image/Transform.hpp"
#include "math/clFMatrix.hpp"
#include "math/clFTensor.hpp"
#include <iostream>

namespace control {

  /**
   * @brief Loads an input set from a directory, with or without classes
   */
  class InputSetLoader {
  public:
    /**
     * @brief Builds a new loader, specifying the dimensions  of the data to be loaded
     * @param tensor_size The tensor depth to use for storing the images
     * @param input_width The width of the input images, rescaling as needed
     * @param input_height The height of the input images, rescaling as needed
     */
    InputSetLoader(size_t tensor_size, size_t input_width, size_t input_height)
        : tensor_size(tensor_size), input_width(input_width), input_height(input_height) {}


    /**
     * @brief Load a directory of images
     * @param path The path of the top directory to load
     * @param load_classes If true, the classes will be loaded as well. Note that this requires a
     * specific directory structure :
     * /root_directory/
     *   /class_1/
     *     /image_1.png
     *     /image_2.png
     *  /class_2/
     *     /image_3.png
     *  ...
     *
     *  Otherwise, every images will be loaded recursively
     * @param shuffle_samples
     * @return The loaded input set
     */
    InputSet load(const std::filesystem::path &path, bool load_classes, bool shuffle_samples) const;

    size_t getTensorSize() const { return tensor_size; }
    size_t getInputWidth() const { return input_width; }
    size_t getInputHeight() const { return input_height; }

    /**
     * @brief Return the pre process engine used to transform the images before rescaling
     * @return A reference to the pre process engine
     */
    image::transform::TransformEngine &getPreProcessEngine() { return preprocess_engine; }
    /**
     * @brief Return the post process engine used to transform the images after rescaling
     * @return A reference to the post process engine
     */
    image::transform::TransformEngine &getPostProcessEngine() { return postprocess_engine; }

  private:
    InputSet loadWithClasses(const std::filesystem::path &path, bool shuffle_samples) const;
    InputSet loadWithoutClasses(const std::filesystem::path &path, bool shuffle_samples) const;

    image::transform::TransformEngine preprocess_engine, postprocess_engine;
    size_t tensor_size, input_width, input_height;
  };

}   // namespace control
