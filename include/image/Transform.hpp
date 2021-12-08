#pragma once

#include "Image.hpp"

namespace image::transform {

  enum class TransformType {
    crop,
    resize,
    binaryScale,
    inversion,
    equalize,
    restriction,
    binaryScaleByMedian,
  };

  // Transformation base class
  // Note that all transformations are assumed to be reentrant
  class Transformation {
  public:
    virtual ~Transformation() = default;
    virtual bool transform(image::GrayscaleImage &image) = 0;
  };

  class Resize : public Transformation {
  private:
    size_t width, height;

  public:
    Resize(size_t width, size_t height);
    bool transform(image::GrayscaleImage &image) override;
  };

  class Crop : public Transformation {
  private:
    size_t width, height, orig_x, orig_y;

  public:
    Crop(size_t width, size_t height, size_t orig_x = 0, size_t orig_y = 0);
    bool transform(image::GrayscaleImage &image) override;
  };

  class Restriction : public Transformation {
  private:
    size_t desired_step;

  public:
    Restriction(size_t desired_step = 1);
    bool transform(image::GrayscaleImage &image) override;
  };

  class Equalize : public Transformation {
  public:
    bool transform(image::GrayscaleImage &image) override;
  };

  class BinaryScale : public Transformation {
  public:
    bool transform(image::GrayscaleImage &image) override;
  };

  class BinaryScaleByMedian : public Transformation {
  public:
    bool transform(image::GrayscaleImage &image) override;
  };

  class Inversion : public Transformation {
  public:
    bool transform(image::GrayscaleImage &image) override;
  };

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