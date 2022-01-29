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
    void append(const math::FloatMatrix &image) { inputs.push_back(image); }

    /** Move copy a single image to the set
     *
     * @param image
     */
    void append(math::FloatMatrix &&image) { inputs.push_back(image); }

    /** Appends a single image to the set, first convert it to a float matrix
     *
     * @param image
     * @param normalization Divide every pixel by this value
     */
    void append(const image::GrayscaleImage &image, float normalization = 255.0f) {
      inputs.push_back(image::imageToMatrix(image, normalization));
    }
  };

  /** @brief Helper class to load an ImageInputSet from a directory
   *
   */
  class ImageInputSetLoader {
  public:
    /** Load a set of images from a directory or a single file
     *  If path is a directory, images will be loaded recursively
     *
     * @param path
     * @return
     */
    static ImageInputSet load(const std::filesystem::path &path);

    /** Load a set of images from a directory or a single file, and records their paths in the
     * process
     * If path is a directory, images will be loaded recursively
     *
     * @param path
     * @return
     */
    static std::pair<ImageInputSet, std::vector<std::filesystem::path>>
    loadWithPaths(const std::filesystem::path &path);

  private:
    static ImageInputSet loadFile(const std::filesystem::path &path,
                                  std::vector<std::filesystem::path> *paths);

    static ImageInputSet loadDirectory(const std::filesystem::path &path,
                                       std::vector<std::filesystem::path> *paths);

    static void loadDirectory(const std::filesystem::path &path, ImageInputSet &set,
                              std::vector<std::filesystem::path> *paths);
  };

}   // namespace control
