#include "CNNModel.hpp"


namespace cnnet {

  CNNModel::CNNModel() {
    cnn = std::make_unique<CNN>();
    mlp = std::make_unique<MLPerceptron>();
  }

  // CNNModel::CNNModel(CNNModel &&other) {}


  math::FloatMatrix CNNModel::predict(math::FloatMatrix const &input) {
    cnn->predict(input, flatten);
    std::cout << flatten << std::endl;
    std::cout << *mlp << std::endl;
    return mlp->predict(flatten);
  }


  CNNModel CNNModelFactory::random(CNNTopology const &topoCNN, MLPTopology &topoMLP) {
    CNNModel res;

    auto &cnn = res.getCnn();
    cnn.setTopology(topoCNN);
    cnn.setActivationFunction(af::ActivationFunctionType::relu);
    cnn.randomizeWeight();

    const size_t size = cnn.getOutputSize();
    res.flatten = FloatMatrix(size, 1);
    topoMLP.push_front(size);

    auto &mlp = res.getMlp();
    mlp.setTopology(topoMLP);
    mlp.setActivationFunction(af::ActivationFunctionType::sigmoid);

    return std::move(res);
  }

}   // namespace cnnet
