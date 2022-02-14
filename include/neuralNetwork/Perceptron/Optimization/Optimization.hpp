#pragma once

#include <vector>

#include "BackpropStorage.hpp"
#include "MLPerceptron.hpp"
#include "Matrix.hpp"


namespace nnet {

  /**
   * @brief Base class for all optimization
   */
  class Optimization {
  public:
    Optimization() = default;
    virtual ~Optimization() = default;

    /**
     *
     * @param storage
     */
    virtual void optimize(BackpropStorage &storage) = 0;

    /**
     * @brief Some optimization require to be updated after each epoch
     * We provide an empty declaration here to avoid to have to implement it where not needed
     */
    virtual void update(){};
  };

  /**
   * @brief Stochastic gradient descent optimization without momentum
   */
  class SGDOptimization : public Optimization {
  public:
    explicit SGDOptimization(const MLPerceptron &perceptron, float lr);

    void optimize(BackpropStorage &storage) override;

  private:
    const float learning_r;
  };

  /**
   * @brief SGD variant using a decaying learning rate
   */
  class DecayOptimization : public Optimization {
  public:
    DecayOptimization(const MLPerceptron &perceptron, const float lr_0, const float dr)
        : initial_lr(lr_0), decay_r(dr), learning_r(lr_0), epoch(0) {}

    void optimize(BackpropStorage &storage) override;
    void update() override;

  private:
    const float initial_lr;
    const float decay_r;
    float learning_r;

    size_t epoch = 0;
  };

  /**
   * @brief SGD Variant using momentum
   */
  class MomentumOptimization : public Optimization {
  public:
    MomentumOptimization(const MLPerceptron &perceptron, float learning_rate, float momentum);

    void optimize(BackpropStorage &storage) override;

  private:
    const float lr;
    const float momentum;
    std::vector<math::FloatMatrix> old_weight_change;
  };

  /**
   * @brief Combined Decay and Momentum optimization
   */
  class DecayMomentumOptimization : public Optimization {
  public:
    DecayMomentumOptimization(const MLPerceptron &perceptron, float lr_0, float dr, float mom);

    void optimize(BackpropStorage &storage) override;
    void update() override;

  private:
    const float initial_lr;
    const float decay_r;
    float learning_r;
    const float momentum;

    size_t epoch = 0;

    std::vector<math::FloatMatrix> old_weight_change;
  };

  /**
   * @brief RProp optimization
   */
  class RPropPOptimization : public Optimization {
  public:
    explicit RPropPOptimization(const MLPerceptron &perceptron, float eta_p = 1.2,
                                float eta_m = 0.5, float lr_max = 50.0, float lr_min = 1e-6);

    void optimize(BackpropStorage &storage) override;

  private:
    std::vector<math::FloatMatrix> weights_updates;
    std::vector<math::FloatMatrix> old_gradients;
    std::vector<math::FloatMatrix> weights_changes;
    const float eta_plus;
    const float eta_minus;
    const float update_max;
    const float update_min;
  };
}   // namespace nnet
