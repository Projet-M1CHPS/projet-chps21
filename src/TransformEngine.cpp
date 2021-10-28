#include <fstream>
#include <iostream>
#include <map>

#include "Transform.hpp"
namespace image::transform {

namespace transform_enumerates {

const std::map<std::string, TRANSFORM_ENUM> transformEnumMap(
    {{"greyscale", TRANSFORM_ENUM::GREYSCALE},
     {"binaryscale", TRANSFORM_ENUM::BINARYSCALE},
     {"histograminversion", TRANSFORM_ENUM::HISTOGRAMINVERSION},
     {"histogramspread", TRANSFORM_ENUM::HISTOGRAMSPREAD},
     {"histogrambinaryscale", TRANSFORM_ENUM::HISTOGRAMBINARYSCALE},
     // Add new functions here
     {"_", TRANSFORM_ENUM::NOTRANSFORM}});
static TRANSFORM_ENUM strToTransformEnum(std::string identifier) {
    TRANSFORM_ENUM t_enum = transformEnumMap.find(identifier)->second;
    if (t_enum == NOTRANSFORM)
        std::cout
            << "/!\\ " << identifier
            << "is not recognised as a valid Transform. \nIf you declared a "
               "new transformation please declare it in: TransformEngine.cpp "
               "[namespace 'transform_enumerates'] subfunctions."
            << std::endl;
    return t_enum;
}

static std::shared_ptr<Transformation> getTransformationFromString(
    std::string identifier) {
    switch (transform_enumerates::strToTransformEnum(identifier)) {
        case GREYSCALE:
            return std::make_shared<GreyScale>();
        case BINARYSCALE:
            return std::make_shared<BinaryScale>();
        case HISTOGRAMINVERSION:
            return std::make_shared<HistogramInversion>();
        case HISTOGRAMBINARYSCALE:
            return std::make_shared<HistogramBinaryScale>();
        case HISTOGRAMSPREAD:
            return std::make_shared<HistogramSpread>();
        // Add new functions here
        default:
            return std::make_shared<NoTransform>();
    }
}

static std::string transformEnumToStr(TRANSFORM_ENUM t_enum) {
    for (auto pair : transformEnumMap)
        if (pair.second == t_enum) return pair.first;
}

}  // namespace transform_enumerates

void TransformEngine::loadFromFile(std::string const &fileName) {
    std::ifstream fp(fileName);
    transformationsEnums.clear();
    transformations.clear();
    for (std::string line; std::getline(fp, line, '\n');) {
        transformationsEnums.push_back(
            transform_enumerates::strToTransformEnum(line));
        transformations.push_back(
            transform_enumerates::getTransformationFromString(line));
    }
    fp.close();
}
void TransformEngine::saveToFile(std::string const &fileName) const {
    std::ofstream fp(fileName);
    for (TRANSFORM_ENUM each : transformationsEnums)
        fp << transform_enumerates::transformEnumToStr(each) << '\n';
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

Image TransformEngine::transform(Image const &image) {
    std::cout << "> TransformEngine::transform()" << std::endl;
    Image copy = image;
    for (std::shared_ptr<Transformation> tr : transformations)
        tr->transform(copy);
    std::cout << "< TransformEngine::transform()" << std::endl;
    return copy;
}

}  // namespace image::transform
