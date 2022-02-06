#pragma once

#include "CNN.hpp"
#include "CNNTopology.hpp"
#include "neuralNetwork/Model/Model.hpp"


namespace cnnet {
  class CNNModel : public Model {
  public:
    CNNModel();
    ~CNNModel() = default;

    CNNModel(const CNNModel &) = delete;
    CNNModel(CNNModel &&) noexcept = default;

    math::FloatMatrix predict(math::FloatMatrix const &input) override {
      return cnn->predict(input);
    }

    [[nodiscard]] CNN &getCnn() { return *cnn; }
    [[nodiscard]] CNN const &getCnn() const { return *cnn; }

  private:
    std::unique_ptr<CNN> cnn;
  };

  class CNNModelFactory {
  public:
    CNNModelFactory() = delete;

    static std::unique_ptr<CNNModel> create(CNNTopology const &topology) {
      auto res = std::make_unique<CNNModel>();
      auto &cnn = res->getCnn();
      cnn.setTopology(topology);
      cnn.setActivationFunction(af::ActivationFunctionType::relu);
      cnn.randomizeWeight();
      return res;
    }
  };
}   // namespace cnnet