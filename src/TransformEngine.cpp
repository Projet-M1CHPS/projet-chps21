#include <cassert>
#include <iostream>

#include "Transform.hpp"

namespace image::transform {

void TransformEngine::insertTransformation(
    size_t position, std::shared_ptr<Transformation> transformation) {
    transformations.insert(transformations.begin() + position, transformation);
}

void TransformEngine::addTransformation(
    std::shared_ptr<Transformation> transformation) {
    transformations.push_back(transformation);
}

Image TransformEngine::transform(Image const &image) {
    Image copy(image);
    for (std::shared_ptr<Transformation> tr : transformations)
        tr->transform(copy);
    return copy;
}

}  // namespace image::transform
