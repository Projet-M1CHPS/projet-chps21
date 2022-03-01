#pragma once

#include "Matrix.hpp"
#include "image/Image.hpp"
#include "inputSet.hpp"
#include <filesystem>

namespace control {

  /** @brief A set of images to be used as inputs to the control system.
   *
   */
  class ImageInputSet : public InputSet {
  public:
    ImageInputSet() = default;

    /** Appends a single image to the set by copying it
     *
     * @param image
     */
    void append(const math::clFMatrix &image, utils::clWrapper &wrapper, cl::CommandQueue &queue,
                bool blocking = true) {
      inputs.emplace_back(image, wrapper, queue, blocking);
    }

    void append(const math::clFMatrix &image, utils::clWrapper &wrapper, bool blocking = true) {
      append(image, wrapper, wrapper.getDefaultQueue(), blocking);
    }

    void append(math::clFMatrix&& image) {
      inputs.emplace_back(std::move(image));
    }

    static ImageInputSet load(const std::filesystem::path &source,
                                           utils::clWrapper &wrapper, cl::CommandQueue &queue,
                                           bool blocking = true);

    static ImageInputSet load(const std::filesystem::path &directory,
                                           utils::clWrapper &wrapper, bool blocking = true) {
      return load(directory, wrapper, wrapper.getDefaultQueue(), blocking);
    }
  };
}   // namespace control
