#pragma once

namespace image::transform {

// Transformation base class
// Note that all transformations are assumed to be reentrant
class Transformation {
public:
  virtual ~Transformation() = default;
  virtual void transform(image::Image &image) = 0;

private:
};

class GreyScale : public Transformation {
public:
  virtual void transform(image::Image &image) override;

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

  // Add the transformation at the given position in the list
  void addTransformation(std::shared_ptr<Transformation> transformation, size_t position);

  // Add the transformation at the end of the transformation list
  // (May be renamed push_back() ?)
  void addTransformation(std::shared_ptr<Transformation> transformation);
  
private:
  // Transformations should be applied in order
  std::vector<std::shared_ptr<Transformation>> transformations;
};

} // namespace image::transform