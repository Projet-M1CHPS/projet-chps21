#pragma once

#include <vector>

#include "BackpropStorage.hpp"
#include "Matrix.hpp"


namespace nnet {

  /** Keep this in sync with new classes
   *
   */
  enum class TrainingAlgorithm { standard, decay, momentum, rpropPlus };

  template<typename T>
  class TrainingMethod {
  public:
    TrainingMethod() = default;
    virtual ~TrainingMethod() = default;

    virtual void compute(BackpropStorage<T> &storage) = 0;
  };

  template<typename T>
  class StandardTrainingMethod : public TrainingMethod<T> {
  public:
    StandardTrainingMethod(const T learningRate) : learningRate(learningRate) {}
    virtual ~StandardTrainingMethod() = default;

    void compute(BackpropStorage<T> &storage) {
      storage.weights->at(storage.index) -= (storage.gradient * learningRate);
    }

  private:
    const T learningRate;
  };


  template<typename T>
  class DecayTrainingMethod : public TrainingMethod<T> {
  public:
    DecayTrainingMethod(const T lr_0, const T dr)
        : learningRate_0(lr_0), decayRate(dr), learningRate(lr_0), epoch(0) {}
    virtual ~DecayTrainingMethod() = default;

    void compute(BackpropStorage<T> &storage) {
      storage.weights->at(storage.index) -= (storage.gradient * learningRate);
    }

    void incrEpoch() {
      epoch++;
      learningRate = (1 / (1 + decayRate * epoch)) * static_cast<T>(learningRate_0);
    }

  private:
    const T learningRate_0;
    const T decayRate;
    T learningRate;

    size_t epoch = 0;
  };


  template<typename T>
  class MomentumTrainingMethod : public TrainingMethod<T> {
  public:
    MomentumTrainingMethod(const std::vector<size_t> &topology, const T learningRate,
                           const T momentum)
        : learningRate(learningRate), momentum(momentum) {
      for (size_t i = 0; i < topology.size() - 1; i++) {
        dw_old.push_back(math::Matrix<T>(topology[i + 1], topology[i]));
      }
    }
    virtual ~MomentumTrainingMethod() = default;

    void compute(BackpropStorage<T> &storage) {
      auto dw = (storage.gradient * learningRate) + (dw_old[storage.index] * momentum);
      storage.weights->at(storage.index) -= dw;
      dw_old[storage.index] = std::move(dw);
    }

  private:
    T learningRate;
    T momentum;
    std::vector<math::Matrix<T>> dw_old;
  };


  template<typename T>
  class RPropPTrainingMethod : public TrainingMethod<T> {
  public:
    RPropPTrainingMethod(const std::vector<size_t> &topology, const T eta_p = 1.2,
                         const T eta_m = 0.5, const T lr_max = 50.0, const T lr_min = 1e-6)
        : eta_plus(eta_p), eta_minus(eta_m), lr_max(lr_max), lr_min(lr_min) {
      for (size_t i = 0; i < topology.size() - 1; i++) {
        lr_old.push_back(math::Matrix<T>(topology[i + 1], topology[i]));
        gradient_old.push_back(math::Matrix<T>(topology[i + 1], topology[i]));
      }
      for (auto &i : lr_old) { i.fill(0.1); }
      for (auto &i : gradient_old) { i.fill(1.0); }
    }
    virtual ~RPropPTrainingMethod() = default;


    void compute(BackpropStorage<T> &storage) {
      auto &weights = storage.weights->at(storage.index);
      for (size_t i = 0; i < weights.getRows(); i++) {
        for (size_t j = 0; j < weights.getCols(); j++) {
          T dw = 0.0;
          if (storage.gradient(i, j) * gradient_old[storage.index](i, j) > 0.0) {
            lr_old[storage.index](i, j) = std::min(eta_plus * lr_old[storage.index](i, j), lr_max);
            if (storage.gradient(i, j) > 0) dw = -1 * lr_old[storage.index](i, j);
            else
              dw = lr_old[storage.index](i, j);
            weights(i, j) = weights(i, j) + dw;
          } else if (storage.gradient(i, j) * gradient_old[storage.index](i, j) < 0.0) {
            if (gradient_old[storage.index](i, j) > 0) dw = -1 * lr_old[storage.index](i, j);
            else
              dw = lr_old[storage.index](i, j);
            weights(i, j) = weights(i, j) - dw;
            lr_old[storage.index](i, j) = std::max(eta_minus * lr_old[storage.index](i, j), lr_min);
            storage.gradient(i, j) = 0.0;
          } else {
            if (storage.gradient(i, j) > 0) dw = -1 * lr_old[storage.index](i, j);
            else
              dw = lr_old[storage.index](i, j);
            weights(i, j) = weights(i, j) + dw;
          }
          gradient_old[storage.index](i, j) = storage.gradient(i, j);
        }
      }
    }

  private:
    std::vector<math::Matrix<T>> lr_old;
    std::vector<math::Matrix<T>> gradient_old;
    const T eta_plus;
    const T eta_minus;
    const T lr_max;
    const T lr_min;
  };

}   // namespace nnet
