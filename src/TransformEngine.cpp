#include <iostream>
#include <map>

#include "Transform.hpp"
namespace image::transform {

namespace transform_enumerates {

enum TRANSFORM_ENUM {
    GREYSCALE,
    BINARYSCALE,
    HISTOGRAMINVERSION,
    HISTOGRAMBINARYSCALE,
    NOTRANSFORM
};

static TRANSFORM_ENUM strToTransformEnum(std::string identifier) {
    TRANSFORM_ENUM t_enum = std::map<std::string, TRANSFORM_ENUM>(
                                {{"greyscale", GREYSCALE},
                                 {"binaryscale", BINARYSCALE},
                                 {"histograminversion", HISTOGRAMINVERSION},
                                 {"histograminversion", HISTOGRAMINVERSION},
                                 {"histogrambinaryscale", HISTOGRAMBINARYSCALE},
                                 // Add new functions here
                                 {"_", NOTRANSFORM}})
                                // Not here !
                                .find(identifier)
                                ->second;
    if (t_enum == NOTRANSFORM)
        std::cout
            << "/!\\ " << identifier
            << "is not recognised as a valid Transform. \nIf you declared a "
               "new transformation please declare it in: TransformEngine.cpp "
               "[namespace 'transform_enumerates'] subfunctions."
            << std::endl;
    return t_enum;
}

static std::string TransformEnumToStr(TRANSFORM_ENUM t_enum) {
    switch (t_enum) {
        case GREYSCALE:
            return "greyscale";
        case BINARYSCALE:
            return "binaryscale";
        case HISTOGRAMINVERSION:
            return "histograminversion";
        case HISTOGRAMBINARYSCALE:
            return "histogrambinaryscale";
        default:
            return "_";
    }
}

static std::shared_ptr<Transformation> _getTransformationFromString(
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
        default:
            return std::make_shared<NoTransform>();
    }
}

}  // namespace transform_enumerates

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
