#include "Image.hpp"
#include "NeuralNetwork.hpp"
#include "Utils.hpp"
#include <iostream>


// Test main, should be replaced by the python interface
int main(int argc, char **argv) {

  // What the main could look like in the end
  /* Load a single image
  image::ImageLoader loader("test.ppm");

  try {
    image::Image img = loader.load();
  } catch (utils::IOException &e) {
    std::cout << "Could not load test image : " << std::endl;
    e.what();
  }

  // Add a simple transformation
  image::transform::TransformEngine te;
  te.addTransformation(std::make_shared<image::transform::GreyScale>());
  te.transform(img);

  nnet::NeuralNetwork nn;

  try {
    std::vector<float> res = nn.runOnInput(img.begin(), img.end());

    std::cout << "NN Output: " << std::endl;
    for (int i = 0; i < res.size(); i++) {
      std::cout << res[i] << std::endl;
    }
  } catch (std::runtime_error &e) {
    std::cout << "Could not run neural network : " << std::endl;
    e.what();
  } */

  return 0;
}