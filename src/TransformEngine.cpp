#include <cassert>
#include <iostream>

#include "Transform.hpp"

void image::transform::TransformEngine::insertTransformation(
    size_t position, std::shared_ptr<Transformation> transformation) {
    this->transformations.insert(this->transformations.begin() + position,
                                 transformation);
}

void image::transform::TransformEngine::addTransformation(
    std::shared_ptr<Transformation> transformation) {
    std::cout << "> TransformEngine::addTransformation()" << std::endl;
    this->transformations.push_back(transformation);
    std::cout << "< TransformEngine::addTransformation()" << std::endl;
}

image::Image image::transform::TransformEngine::transform(
    image::Image const &image) {
    std::cout << "> TransformEngine::transform()" << std::endl;
    image::Image copy(image.width, image.height,
                      std::vector<Color>(image.colors));
    image::Image& copy_ref = copy;
    for (size_t i = 0; i < this->transformations.size(); i++) {
        std::cout << ">> Transformation[" << i << "]" << std::endl;
        std::shared_ptr<image::transform::Transformation> each_tr =
            this->transformations[i];
        image::transform::Transformation *fct = each_tr.get();
        fct->transform(copy_ref);
        std::cout << "<< Transformation[" << i << "]" << std::endl;
    }
    std::cout << "< TransformEngine::transform()" << std::endl;
    return copy;
}