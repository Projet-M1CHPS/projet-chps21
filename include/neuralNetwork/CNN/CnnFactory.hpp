#pragma once

#include <sstream>
#include <string>
#include <utility>
#include <vector>


namespace cnnet {

  class CnnTopology {
  public:
    CnnTopology(const std::pair<size_t, size_t> &inputSize);
    ~CnnTopology() = default;

    void addConvolution(const size_t features, const std::pair<size_t, size_t> &filterSize,
                        const size_t stride, const size_t padding);
    void addPooling(const std::pair<size_t, size_t> &poolSize, const size_t stride);

    void getTopology();

  private:
    std::vector<CnnLayer> layers;
  };


  const CnnTopology &stringToTopology(const std::string &str) {
    std::stringstream ss(str);
    size_t inputRow = 0, inputCol = 0;

    ss >> inputRow >> inputCol;
    CnnTopology res({inputRow, inputCol});

    std::string type;
    while (ss >> type) {
      if (type == "convolution") {
        size_t features = 0, filterRow = 0, filterCol = 0, stride = 0, padding = 0;
        ss >> features >> filterRow >> filterCol >> stride >> padding;
        res.addConvolution(features, {filterRow, filterCol}, stride, padding);
      } else if (type == "pooling") {
        size_t poolRow, poolCol, stride;
        ss >> poolRow >> poolCol >> stride;
        res.addPooling({poolRow, poolCol}, stride);
      } else {
        throw std::invalid_argument("Invalid type " + type);
      }
    }
    return res;
  }
}   // namespace cnnet