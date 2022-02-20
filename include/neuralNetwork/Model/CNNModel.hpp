#pragma once

#include "Model.hpp"
#include "neuralNetwork/CNN/CNN.hpp"
#include "neuralNetwork/Perceptron/MLPerceptron.hpp"

namespace cnnet {
  using namespace nnet;
  using namespace math;

  class CNNModel : public Model {
    friend class CNNModelFactory;
  public:
    CNNModel();
    ~CNNModel() = default;

    CNNModel(const CNNModel &) = delete;
    CNNModel(CNNModel &&other) noexcept {
      this->cnn = std::move(other.cnn);
      this->mlp = std::move(other.mlp);
      this->flatten = std::move(other.flatten);
    };

    math::FloatMatrix predict(math::FloatMatrix const &input) override;

    [[nodiscard]] CNN &getCnn() { return *cnn; }
    [[nodiscard]] CNN const &getCnn() const { return *cnn; }

    [[nodiscard]] FloatMatrix &getFlatten() { return flatten; }
    [[nodiscard]] FloatMatrix const &getFlatten() const { return flatten; }

    [[nodiscard]] MLPerceptron &getMlp() { return *mlp; }
    [[nodiscard]] MLPerceptron const &getMlp() const { return *mlp; }

  private:
    std::unique_ptr<CNN> cnn;
    std::unique_ptr<MLPerceptron> mlp;

    FloatMatrix flatten;
  };


  class CNNModelFactory {
  public:
    CNNModelFactory() = delete;

    static CNNModel random(CNNTopology const &topoCNN, MLPTopology &topoMLP);
  };
}   // namespace cnnet