#include <fstream>
#include <iostream>
#include <map>

#include "Transform.hpp"
namespace image::transform {

namespace {

const std::map<std::string, TransformType> transformEnumMap(
    {{"greyscale", TransformType::greyscale},
     {"binaryscale", TransformType::binaryScale},
     {"histograminversion", TransformType::histogramInversion},
     {"histogramspread", TransformType::histogramSpread},
     {"histogrambinaryscale", TransformType::histogramBinaryScale},
     // Add new functions here
     {"", TransformType::noTransform}});

TransformType strToTransformEnum(std::string identifier) {
  TransformType t_enum = transformEnumMap.find(identifier)->second;
  if (t_enum == TransformType::noTransform)
    std::cout << "/!\\ " << identifier
              << "is not recognised as a valid Transform. \nIf you declared a "
                 "new transformation please declare it in: TransformEngine.cpp "
                 "[namespace 'transform_enumerates'] subfunctions."
              << std::endl;
  return t_enum;
}

std::shared_ptr<Transformation>
getTransformationFromString(std::string identifier) {
  switch (strToTransformEnum(identifier)) {
  case TransformType::greyscale:
    return std::make_shared<GreyScale>();
  case TransformType::binaryScale:
    return std::make_shared<BinaryScale>();
  case TransformType::histogramInversion:
    return std::make_shared<HistogramInversion>();
  case TransformType::histogramBinaryScale:
    return std::make_shared<HistogramBinaryScale>();
  case TransformType::histogramSpread:
    return std::make_shared<HistogramSpread>();
  // Add new functions here
  default:
    return std::make_shared<NoTransform>();
  }
}

std::string transformEnumToStr(TransformType t_enum) {
  for (auto pair : transformEnumMap)
    if (pair.second == t_enum)
      return pair.first;
  return "_";
}

} // namespace

void TransformEngine::loadFromFile(std::string const &fileName) {
  std::ifstream fp(fileName);

  // Remove old transformations
  transformationsEnums.clear();
  transformations.clear();

  for (std::string line; std::getline(fp, line, '\n');) {
    transformationsEnums.push_back(strToTransformEnum(line));
    transformations.push_back(getTransformationFromString(line));
  }
  fp.close();
}

void TransformEngine::saveToFile(std::string const &fileName) const {
  std::ofstream fp(fileName);

  for (TransformType each : transformationsEnums)
    fp << transformEnumToStr(each) << '\n';
  fp.close();
}

void TransformEngine::insertTransformation(
    size_t position, std::shared_ptr<Transformation> transformation) {
  transformations.insert(transformations.begin() + position, transformation);
}

void TransformEngine::addTransformation(
    std::shared_ptr<Transformation> transformation) {
  transformations.push_back(transformation);
}

GrayscaleImage TransformEngine::transform(GrayscaleImage const &image) {
  std::cout << "> TransformEngine::transform()" << std::endl;

  GrayscaleImage copy = image;
  for (std::shared_ptr<Transformation> tr : transformations)
    tr->transform(copy);

  std::cout << "< TransformEngine::transform()" << std::endl;
  return copy;
}

} // namespace image::transform
