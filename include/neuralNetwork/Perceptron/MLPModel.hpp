#pragma once

#include "MLPerceptron.hpp"
#include "Model.hpp"

namespace nnet {

  /**
   * @brief A model for a multi-layer perceptron.
   */
  class MLPModel : public Model {
  public:
    MLPModel();
    MLPModel(std::unique_ptr<MLPerceptron> &&perceptron);

    [[nodiscard]] MLPerceptron &getPerceptron() { return *perceptron; }
    [[nodiscard]] MLPerceptron const &getPerceptron() const { return *perceptron; }

    /**
     * @brief Feed the given input to the perceptron
     * @param input The matrix to be fed to the perceptron
     * @return The output of the perceptron
     */
    math::clFMatrix predict(math::clFMatrix const &input) const override;

    math::clFTensor predict(math::clFTensor const &input) const override {
      // TODO : Implement me
      assert(0 && "implement me");
      return {1, 1, 1};
    }

    bool load(const std::filesystem::path &path) override;
    bool save(const std::filesystem::path &path) const override;

    /**
     * @brief Creates a random Model
     * @param seed The seed to be used for the randomization
     * @param topology The topology to be used for the model
     * @return a random model
     */
    static std::unique_ptr<MLPModel>
    random(MLPTopology const &topology,
           af::ActivationFunctionType af = af::ActivationFunctionType::sigmoid);

    /**
     * @brief Creates a random model that uses the relu/sigmoid activation functions
     * The sigmoids are used for clipping the gradient to prevent its explosion
     * TODO: In the long run, this should be replaced by a L2 normalization
     * @param seed The seed to be used for the randomization
     * @param topology The topology to be used for the model
     * @return
     */
    static std::unique_ptr<MLPModel> randomReluSigmoid(MLPTopology const &topology);

  private:
    std::unique_ptr<MLPerceptron> perceptron;
  };
}   // namespace nnet