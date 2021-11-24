#include <fstream>
#include <iostream>
#include <map>

#include "Transform.hpp"
namespace image::transform {

  namespace {

    const std::map<std::string, TransformType> transformEnumMap({
            {"binaryscale", TransformType::binaryScale},
            {"inversion", TransformType::inversion},
            {"binaryScaleByMedian", TransformType::binaryScaleByMedian},
            {"resize", TransformType::resize},
            // Add new functions here
    });

    TransformType strToTransformEnum(std::string identifier) {
      auto t_enum = transformEnumMap.find(identifier);
      if (t_enum == transformEnumMap.end())
        throw "[ERROR]: " + identifier + "is not recognised as a valid Transform. \n" +
                "If you declared a new transformation please declare it in: TransformEngine.cpp " +
                "[namespace 'transform_enumerates'] subfunctions.\n";
      return t_enum->second;
    }

    std::shared_ptr<Transformation> getTransformationFromString(std::string identifier) {
      switch (strToTransformEnum(identifier)) {
        case TransformType::binaryScale:
          return std::make_shared<BinaryScale>();
        case TransformType::inversion:
          return std::make_shared<Inversion>();
        case TransformType::binaryScaleByMedian:
          return std::make_shared<BinaryScaleByMedian>();
        case TransformType::resize:
          return std::make_shared<Resize>(0, 0);   // TODO: Rebuild the TransformationEngine system
        // Add new functions here
        default:
          throw "[ERROR]: " + identifier + "is not recognised as a valid Transform. \n" +
                  "If you declared a new transformation please declare it in: "
                  "TransformEngine.cpp " +
                  "[namespace 'transform_enumerates'] subfunctions.\n";
      }
    }

    std::string transformEnumToStr(TransformType t_enum) {
      for (auto pair : transformEnumMap)
        if (pair.second == t_enum) return pair.first;
      throw "[ERROR]: transformEnumToStr(t_enum): t_enum can not be recognised as a valid "
            "Transform identifier.\nIf you declared a new transformation please declare it in: "
            "TransformEngine.cpp [namespace 'transform_enumerates'] subfunctions.\n";
      return "_";
    }

  }   // namespace

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

  GrayscaleImage TransformEngine::transform(GrayscaleImage const &image) {
    std::cout << "> TransformEngine::transform()" << std::endl;

    GrayscaleImage copy = image;
    for (std::shared_ptr<Transformation> tr : transformations) tr->transform(copy);

    std::cout << "< TransformEngine::transform()" << std::endl;
    return copy;
  }

}   // namespace image::transform
