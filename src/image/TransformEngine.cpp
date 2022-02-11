#include <fstream>
#include <iostream>
#include <map>

#include "Transform.hpp"
namespace image::transform {

  static std::string toLowerString(const std::string &str) {
    std::string copy(str);
    std::for_each(copy.begin(), copy.end(), [](auto &e) { e = std::tolower(e); });
    return copy;
  }

  static const std::map<std::string, TransformType> transformEnumMap({
          {"binaryscale", TransformType::binaryScale},
          {"binaryScaleByMedian", TransformType::binaryScaleByMedian},
          {"resize", TransformType::resize},
          {"crop", TransformType::crop},
          {"inversion", TransformType::inversion},
          {"equalize", TransformType::equalize},
          {"restriction", TransformType::restriction},
          // Add new functions here
  });

  TransformType TransformEngine::strToTransformEnum(std::string identifier) {
    auto t_enum = transformEnumMap.find(toLowerString(identifier));
    if (t_enum == transformEnumMap.end())
      throw std::invalid_argument(
              "[ERROR]: " + toLowerString(identifier) + " is not recognised as a valid Transform. \n" +
              "If you declared a new transformation please declare it in: TransformEngine.cpp " +
              "[namespace 'transform_enumerates'] subfunctions.\n");
    return t_enum->second;
  }

  std::shared_ptr<Transformation>
  TransformEngine::getTransformationFromString(std::string identifier) {
    switch (strToTransformEnum(toLowerString(identifier))) {
      case TransformType::binaryScale:
        return std::make_shared<BinaryScale>();
      case TransformType::binaryScaleByMedian:
        return std::make_shared<BinaryScaleByMedian>();
      case TransformType::resize:   // TODO: Rebuild the TransformationEngine system
        return std::make_shared<Resize>(0, 0);
      case TransformType::crop:   // TODO: Rebuild the TransformationEngine system
        return std::make_shared<Crop>(0, 0, 0, 0);
      case TransformType::inversion:
        return std::make_shared<Inversion>();
      case TransformType::equalize:
        return std::make_shared<Equalize>();
      case TransformType::restriction:   // TODO: Rebuild the TransformationEngine system
        return std::make_shared<Restriction>(32);

      // Add new functions here
      default:
        throw std::invalid_argument("[ERROR]: " + toLowerString(identifier) +
                                    " is not recognised as a valid Transform. \n" +
                                    "If you declared a new transformation please declare it in: "
                                    "TransformEngine.cpp " +
                                    "[namespace 'transform_enumerates'] subfunctions.\n");
    }
  }

  std::string TransformEngine::transformEnumToStr(TransformType t_enum) {
    for (auto pair : transformEnumMap)
      if (pair.second == t_enum) return pair.first;
    throw std::invalid_argument(
            "[ERROR]: transformEnumToStr(t_enum): t_enum can not be recognised as a valid "
            "Transform identifier.\nIf you declared a new transformation please declare it in: "
            "TransformEngine.cpp [namespace 'transform_enumerates'] subfunctions.\n");
    return "_";
  }


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

    for (TransformType each : transformationsEnums) fp << transformEnumToStr(each) << '\n';
    fp.close();
  }

  void TransformEngine::insertTransformation(size_t position,
                                             std::shared_ptr<Transformation> transformation) {
    transformations.insert(transformations.begin() + position, transformation);
  }

  void TransformEngine::addTransformation(std::shared_ptr<Transformation> transformation) {
    transformations.push_back(transformation);
  }

  GrayscaleImage TransformEngine::transform(GrayscaleImage const &image) const {
    GrayscaleImage copy = image;
    for (const std::shared_ptr<Transformation> &tr : transformations) tr->transform(copy);

    return copy;
  }

  void TransformEngine::apply(GrayscaleImage &image) const {
    for (const std::shared_ptr<Transformation> &tr : transformations) tr->transform(image);
  }

}   // namespace image::transform
