#pragma once

#include "Image.hpp"

namespace image::transform {

  /** Enumeration of all supported transformations, used for serialization
   *
   */
  enum class TransformType {
    crop,
    resize,
    binaryScale,
    inversion,
    equalize,
    restriction,
    binaryScaleByMedian,
  };

  /** @brief Interface for GrayscaleImage transformations
   *
   */
  class Transformation {
  public:
    virtual ~Transformation() = default;
    virtual bool transform(image::GrayscaleImage &image) const = 0;
  };

  /** @brief Resizing transformation for GrayscaleImage
   *
   */
  class Resize : public Transformation {
  private:
    size_t width, height;

  public:
    /**
     * Resize the given image to the desired dimensions.
     *
     * This transformations implies losses. We can not assure that a double-resize
     * image will be the same.
     */
    Resize(size_t width, size_t height);
    bool transform(image::GrayscaleImage &image) const override;
  };

  /** @brief Produces a view of a GrayscaleImage
   *
   */
  class Crop : public Transformation {
  private:
    size_t width, height, orig_x, orig_y;

  public:
    /**
     * Transforms the given image into a view of itself from the given origin and with the desired
     * dimensions.
     *
     * Every original image data out the view will be lost.
     *
     * @param orig_x column origin index (0 is left border)
     * @param orig_y row origin index (0 is top border)
     */
    Crop(size_t width, size_t height, size_t orig_x = 0, size_t orig_y = 0);
    bool transform(image::GrayscaleImage &image) const override;
  };


  /** @brief Restriction transformation for GrayscaleImage
   *
   */
  class Restriction : public Transformation {
  private:
    size_t desired_step;

  public:
    /**
     * @brief Add a desired_step between each colors of the given image. The result is an image with
     * less colors and so more "stricts" edges.
     *
     * @param desired_step 1 by default, step between two colors.
     */
    Restriction(size_t desired_step = 1);
    bool transform(image::GrayscaleImage &image) const override;
  };

  /**
   * @brief Equalize the given image colors repartition. (A half-white & half-black image turns into
   * a full-gray image)
   */
  class Equalize : public Transformation {
  public:
    bool transform(image::GrayscaleImage &image) const override;
  };

  /**
   * @brief Apply a Gaussian blur on the given image.
   */
  class Filter : public Transformation {
  public:
    bool transform(image::GrayscaleImage &image) const override;
  };

  /**
   * @brief Transforms every pixels of the given image in black, but edges areas in white.
   *
   * @param image at the end, will be a black & white image.
   */
  class Edges : public Transformation {
  public:
    bool transform(image::GrayscaleImage &image) const override;
  };

  /**
   * @brief Turns the given image in black & white. Based on the mean image color.
   */
  class BinaryScale : public Transformation {
  public:
    bool transform(image::GrayscaleImage &image) const override;
  };

  /**
   * @brief Turns the given image in black & white. Based on the median image color.
   */
  class BinaryScaleByMedian : public Transformation {
  public:
    bool transform(image::GrayscaleImage &image) const override;
  };

  /**
   * @brief Inverts colors for the given image. White turns into black, etc.
   */
  class Inversion : public Transformation {
  public:
    bool transform(image::GrayscaleImage &image) const override;
  };

  /** @brief Pipeline for applying transformations on GrayscaleImage
   *
   */
  class TransformEngine {
  public:
    // We want to be able to store the applied transformations in a file
    // (Should throw an exception on error)
    void loadFromFile(std::string const &fileName);

    // Since we may want to store the transformations alongside other data,
    // this should not rely on eos/eof to know when to stop reading
    void loadFromStream(std::istream &stream);

    void saveToFile(std::string const &fileName) const;
    void saveToStream(std::istream &stream) const;

    // Insert the transformation at the given position in the list
    void insertTransformation(size_t position, std::shared_ptr<Transformation> transformation);

    // Add the transformation at the end of the transformation list
    // (Might be renamed push_back() ?)
    void addTransformation(std::shared_ptr<Transformation> transformation);

    /**
     * @param image  Base image will not be modified.
     * @return a copy of the base image with all transformations applied
     * @brief Apply all transformations of the transformation list for the
     * given image
     */
    [[nodiscard]] image::GrayscaleImage transform(image::GrayscaleImage const &image) const;

    /**
     * @param image  Apply every transformation in-place
     * @brief Apply all transformations of the transformation list for the
     * given image
     */
    void apply(image::GrayscaleImage &image) const;

  private:
    // Keeping the transformations identifiers for I/O easily.
    std::vector<TransformType> transformationsEnums;
    // Transformations should be applied in order
    std::vector<std::shared_ptr<Transformation>> transformations;
  };

}   // namespace image::transform