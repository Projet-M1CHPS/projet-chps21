#pragma once

#include "ActivationFunction.hpp"
#include "Matrix.hpp"
#include "Utils.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#define alpha .1

namespace nnet
{

  /**
 * @brief Enum of the supported floating precision
 *
 * Used for serialization
 *
 */
  enum class FloatingPrecision
  {
    float32,
    float64,
  };

  /**
 * @brief
 *
 * @param str
 * @return FloatingPrecision
 */
  FloatingPrecision strToFPrecision(const std::string &str);

  const char *fPrecisionToStr(FloatingPrecision fp);

  /**
 * @brief Convert a static type to the corresponding enum type
 *
 * @tparam ypename
 * @return FloatingPrecision
 */
  template <typename T>
  FloatingPrecision getFPPrecision()
  {
    if constexpr (std::is_same_v<float, T>)
    {
      return FloatingPrecision::float32;
    }
    else if constexpr (std::is_same_v<double, T>)
    {
      return FloatingPrecision::float64;
    }
    else
    {
      // dirty trick to prevent the compiler for evaluating the else branch
      static_assert(!sizeof(T), "Invalid floating point type");
    }
  }

  /**
 * @brief Base class for the neural network
 *
 */
  class NeuralNetworkBase
  {
  public:
    NeuralNetworkBase() = delete;

    virtual ~NeuralNetworkBase() = default;

    [[nodiscard]] FloatingPrecision getPrecision() const { return precision; }

    virtual void setLayersSize(std::vector<size_t> const &layers) = 0;

    [[nodiscard]] virtual size_t getOutputSize() const = 0;

    [[nodiscard]] virtual size_t getInputSize() const = 0;

    [[nodiscard]] virtual std::vector<size_t> getLayersSize() const = 0;

    virtual void setActivationFunction(af::ActivationFunctionType type) = 0;

    virtual void setActivationFunction(af::ActivationFunctionType af,
                                       size_t layer) = 0;

    [[nodiscard]] virtual const std::vector<af::ActivationFunctionType> &
    getActivationFunctions() const = 0;

    virtual void randomizeSynapses() = 0;

  protected:
    // The precision has no reason to change, so no need for setter method
    // And the ctor can be protected
    explicit NeuralNetworkBase(FloatingPrecision precision) : precision(precision) {}

    FloatingPrecision precision = FloatingPrecision::float32;
  };

  /**
 * @brief A neural network that supports most fp precision as template
 * parameters
 *
 * @tparam real
 */
  template <typename real>
  class NeuralNetwork final : public NeuralNetworkBase
  {
  public:
    using value_type = real;

    /**
     * @brief Construct a new Neural Network object with no layer
     *
     */
    NeuralNetwork() : NeuralNetworkBase(getFPPrecision<real>()) {}

    NeuralNetwork(const NeuralNetwork &other)
        : NeuralNetworkBase(getFPPrecision<real>())
    {
      *this = other;
    }

    NeuralNetwork(NeuralNetwork &&other) noexcept : NeuralNetworkBase(getFPPrecision<real>())
    {
      *this = std::move(other);
    }

    NeuralNetwork &operator=(const NeuralNetwork &) = default;

    NeuralNetwork &operator=(NeuralNetwork &&other) noexcept
    {
      weights = std::move(other.weights);
      biases = std::move(other.biases);
      activation_functions = std::move(other.activation_functions);

      return *this;
    }

    /**
     * @brief Run the network on a set of input, throwing on error
     *
     *
     * @tparam iterator
     * @param begin
     * @param end
     * @return math::Matrix<real> A vector containing the network's output
     */
    template <typename iterator>
    math::Matrix<real> forward(iterator begin, iterator end) const
    {

      const size_t nbInput = std::distance(begin, end);

      if (nbInput != weights.front().getCols())
      {
        throw std::invalid_argument("Invalid number of input");
      }

      math::Matrix<real> current_layer(nbInput, 1);
      std::copy(begin, end, current_layer.begin());

      for (size_t i = 0; i < weights.size(); i++)
      {
        current_layer = forwardOnce(current_layer, i);
      }
      return current_layer;
    }

    template <typename iterator>
    void train(iterator begin_input, iterator end_input, iterator begin_target,
               iterator end_target)
    {
      std::vector<math::Matrix<real>> layers;
      std::vector<math::Matrix<real>> layers_af;
      // std::vector<math::Matrix<real>> errors_buffer;
      const size_t nbInput = std::distance(begin_input, end_input);
      const size_t nbTarget = std::distance(begin_target, end_target);

      if (nbInput != weights.front().getCols() || nbTarget != getOutputSize())
      {
        throw std::invalid_argument("Invalid number of input");
      }

      //
      math::Matrix<real> current_layer(nbInput, 1);
      std::copy(begin_input, end_input, current_layer.begin());

      //
      math::Matrix<real> target(nbTarget, 1);
      std::copy(begin_target, end_target, target.begin());

      // Forward
      // ------------------------------------------------------------------------

      layers.push_back(current_layer);
      layers_af.push_back(current_layer);

      for (size_t i = 0; i < weights.size(); i++)
      {
        // C = W * C + B
        current_layer = weights[i] * current_layer;
        current_layer += biases[i];

        layers.push_back(current_layer);

        // Apply activation function on every element of the matrix
        auto afunc = af::getAFFromType<real>(activation_functions[i]).first;
        std::transform(current_layer.cbegin(), current_layer.cend(),
                       current_layer.begin(), afunc);

        layers_af.push_back(current_layer);
      }

      // Backward
      // ------------------------------------------------------------------------

      // Erreur de l'output
      math::Matrix<real> current_error = target - current_layer;

      // printf("backprop\n");

      for (long i = weights.size() - 1; i >= 0; i--)
      {
        // std::cout << current_error << "\n" << std::endl;
        // std::cout << "\ni = " << i << std::endl;

        // calcul de S
        math::Matrix<real> gradient(layers[i + 1]);
        // std::cout << "gradient = \n" << gradient << std::endl;
        auto dafunc = af::getAFFromType<real>(activation_functions[i]).second;
        std::transform(gradient.cbegin(), gradient.cend(), gradient.begin(),
                       dafunc);
        // std::cout << "gradient = \n" << gradient << std::endl;

        // std::cout << "eror\n" << current_error << std::endl;
        // calcul de (S * E) * alpha
        gradient.hadamardProd(current_error);
        gradient = gradient * alpha;
        // std::cout << "gradient = \n" << gradient << std::endl;

        // calcul de ((S * E) * alpha) * Ht
        // math::Matrix<real> ht = ;
        math::Matrix<real> delta_weight = gradient * layers_af[i].transpose();
        // std::cout << "delta_weight = \n" << delta_weight << std::endl;

        weights[i] = weights[i] + delta_weight;
        biases[i] = biases[i] + gradient;

        // std::cout << "\ni = " << i << std::endl;
        // std::cout << "weight" << std::endl;
        // std::cout << weights[i] << std::endl;
        // std::cout << "biais" << std::endl;
        // std::cout << biases[i] << std::endl;

        current_error = weights[i].transpose() * current_error;
      }
    }

    template <typename iterator>
    void train_bis(iterator begin_input, iterator end_input, iterator begin_target,
                   iterator end_target, const real learning_rate)
    {
      std::vector<math::Matrix<real>> layers;
      std::vector<math::Matrix<real>> layers_af;
      std::vector<math::Matrix<real>> errors;
      errors.resize(weights.size());

      const size_t nbInput = std::distance(begin_input, end_input);
      const size_t nbTarget = std::distance(begin_target, end_target);

      if (nbInput != weights.front().getCols() || nbTarget != getOutputSize())
      {
        throw std::invalid_argument("Invalid number of input");
      }

      //
      math::Matrix<real> current_layer(nbInput, 1);
      std::copy(begin_input, end_input, current_layer.begin());

      //
      math::Matrix<real> target(nbTarget, 1);
      std::copy(begin_target, end_target, target.begin());

      // Forward
      // ------------------------------------------------------------------------

      layers.push_back(current_layer);
      layers_af.push_back(current_layer);

      for (size_t i = 0; i < weights.size(); i++)
      {
        // C = W * C + B
        current_layer = weights[i] * current_layer + biases[i];

        layers.push_back(current_layer);

        // Apply activation function on every element of the matrix
        auto afunc = af::getAFFromType<real>(activation_functions[i]).first;
        std::transform(current_layer.cbegin(), current_layer.cend(),
                       current_layer.begin(), afunc);

        layers_af.push_back(current_layer);
      }

      // Backward

      // Error
      math::Matrix<real> current_error = target - current_layer;
      errors[errors.size() - 1] = current_error;

      for (long i = weights.size() - 2; i >= 0; i--)
      {
        current_error = weights[i + 1].transpose() * current_error;
        errors[i] = current_error;
      }

      for (long i = weights.size() - 1; i >= 0; i--)
      {
        // calcul de S * (1 - S)
        math::Matrix<real> gradient(layers[i + 1]);
        auto dafunc = af::getAFFromType<real>(activation_functions[i]).second;
        std::transform(gradient.cbegin(), gradient.cend(), gradient.begin(), dafunc);

        // calcul de (S * E) * alpha
        gradient.hadamardProd(errors[i]);
        gradient *= learning_rate;

        // calcul de ((S * E) * alpha) * Ht
        math::Matrix<real> delta_weight = gradient * layers_af[i].transpose();

        weights[i] += delta_weight;
        biases[i] += gradient;
      }
    }

    template <typename iterator>
    math::Matrix<real> predict(iterator begin, iterator end) const
    {
      const size_t nbInput = std::distance(begin, end);

      if (nbInput != weights.front().getCols())
      {
        throw std::invalid_argument("Invalid number of input");
      }

      math::Matrix<real> current_layer(nbInput, 1);
      std::copy(begin, end, current_layer.begin());

      for (size_t i = 0; i < weights.size(); i++)
      {
        current_layer = forwardOnce(current_layer, i);
      }
      return current_layer;
    }

    /**
     * @brief Take a vector of sizes correspondig to the number of neurons
     * in each layer and build the network accordingly. Note that weights are not
     * initialized after this.
     *
     * @param layers
     */
    void setLayersSize(std::vector<size_t> const &layers) override
    {

      if (layers.size() < 2)
      {
        throw std::invalid_argument("Requires atleast 2 layers");
      }

      weights.clear();
      biases.clear();
      for (size_t i = 0; i < layers.size() - 1; i++)
      {
        // Create a matrix of size (layers[i + 1] x layers[i])
        // So that each weight matrix can be multiplied by the previous layer
        weights.push_back(math::Matrix<real>(layers[i + 1], layers[i]));
        biases.push_back(math::Matrix<real>(layers[i + 1], 1));
        activation_functions.push_back(af::ActivationFunctionType::sigmoid);
      }
    }

    [[nodiscard]] size_t getOutputSize() const override
    {
      // The last operation is <Mm x Mn> * <Om * On>
      // So the output is <Mm * On> where On is 1

      if (weights.empty())
      {
        return 0;
      }

      return weights.back().getRows();
    }

    [[nodiscard]] size_t getInputSize() const override
    {
      // The first operation is <Mm x Mn> * <Om * On>
      // So the input is of size <Mn * On> where On is 1
      if (weights.empty())
      {
        return 0;
      }

      return weights.front().getCols();
    }

    [[nodiscard]] std::vector<size_t> getLayersSize() const override
    {
      std::vector<size_t> res(weights.size());

      for (auto w : weights)
      {
        res.push_back(w.getRows());
      }
      return res;
    }

    void setActivationFunction(af::ActivationFunctionType type) override
    {

      for (auto &activation_function : activation_functions)
      {
        activation_function = type;
      }
    }

    void setActivationFunction(af::ActivationFunctionType af,
                               size_t layer) override
    {
      if (layer >= weights.size())
      {
        throw std::invalid_argument("Invalid layer");
      }

      activation_functions[layer] = af;
    }

    [[nodiscard]] const std::vector<af::ActivationFunctionType> &
    getActivationFunctions() const override
    {
      return activation_functions;
    }

    /**
     * @brief Randomizes the weights and biases of the network
     *
     * @param seed
     */
    void randomizeSynapses() override
    {
      for (auto &layer : weights)
      {
        utils::random::randomize<real>(layer, 0, 1);
      }

      for (auto &layer : biases)
      {
        utils::random::randomize<real>(layer, 0, 1);
      }
    }

    [[nodiscard]] std::vector<math::Matrix<real>> &getWeights() { return weights; }

    [[nodiscard]] const std::vector<math::Matrix<real>> &getWeights() const { return weights; }

    [[nodiscard]] std::vector<math::Matrix<real>> &getBiases() { return biases; }

    [[nodiscard]] const std::vector<math::Matrix<real>> &getBiases() const { return biases; }

  private:
    math::Matrix<real> forwardOnce(const math::Matrix<real> &mat,
                                   const size_t index) const
    {
      // C = W * C + B

      math::Matrix<real> res = weights[index] * mat;
      // Avoid a copy by using the += operator
      res += biases[index];

      // Apply activation function on every element of the matrix
      auto afunc = af::getAFFromType<real>(activation_functions[index]).first;
      std::transform(res.cbegin(), res.cend(), res.begin(), afunc);
      return res;
    }

  private:
    std::vector<math::Matrix<real>> weights;
    std::vector<math::Matrix<real>> biases;

    // We want every layer to have its own activation function
    std::vector<af::ActivationFunctionType> activation_functions;
  };

  //std::ostream& operator<<(std::ostream& os, const Pair<T, U>& p)
  template <typename T>
  std::ostream &operator<<(std::ostream &os, const NeuralNetwork<T> &nn)
  {
    const size_t size = nn.getWeights().size();
    os << "-------input-------\n";
    for (size_t i = 0; i < size; i++)
    {
      os << "-----weight[" << i << "]-----\n";
      os << nn.getWeights()[i];
      os << "------bias[" << i << "]------\n";
      os << nn.getBiases()[i];
      if (i != size - 1)
      {
        os << "-----hidden[" << i << "]-----\n";
      }
    }
    os << "-------output------\n";
    return os;
  }
} // namespace nnet