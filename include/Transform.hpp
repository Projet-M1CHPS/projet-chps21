#pragma once

#include "Image.hpp"

namespace image::transform {

// Transformation base class
// Note that all transformations are assumed to be reentrant
class Transformation {
public:
  virtual ~Transformation() = default;
  virtual image::Image* transform(image::Image const &image) = 0;

private:
};

class GreyScale : public Transformation {
public:
  image::Image* transform(image::Image const &image) override;

private:
};

class BlackWhiteScale : public Transformation {
public:
  image::Image* transform(image::Image const &image) override;

private:
};

class TransformEngine {
public:
  // We want to be able to store the applied transformations in a file
  // (Should throw an exception on error)
  void loadFromFile(std::string const &fileName);

  // Since we may want to store the transformations alongside other data,
  // this sould not rely on eos/eof to know when to stop reading
  void loadFromStream(std::istream &stream);

  void saveToFile(std::string const &fileName) const;
  void saveToStream(std::istream &stream) const;

  // Insert the transformation at the given position in the list
  void insertTransformation(size_t position, std::shared_ptr<Transformation> transformation);

  // Add the transformation at the end of the transformation list
  // (May be renamed push_back() ?)
  void addTransformation(std::shared_ptr<Transformation> transformation);
  
  /**
   * @param image  Base image, will not be modified.
   * @return a copy of the base image with all transformations applied
   * @brief Apply all transformations of the transformation list for the given image
  */
  image::Image* transform(image::Image const &image);
private:
  // Transformations should be applied in order
  std::vector<std::shared_ptr<Transformation>> transformations;
};

} // namespace image::transform