#pragma once

#include "MLPModel.hpp"
#include "MLPerceptron.hpp"
#include "Optimization.hpp"
#include "Optimizer.hpp"
#include "neuralNetwork/OptimizationScheduler/OptimizationScheduler.hpp"
#include <iostream>
#include <utility>

namespace nnet {

  /**
   * @brief An optimizer for MLP models, usable with the scheduler module
   */
  class MLPOptimizer : public Optimizer {
  public:
    class WeightUpdateCache;

    /**
     * @brief A delegate class to run the optimizer
     */
    class Operation : public Optimizer::Operation {
    public:
      explicit Operation(MLPOptimizer &optimizer) : optimizer(&optimizer) {}

      /**
       * @brief Runs the optimizer using the cache associated with the thread rank
       * @param thread_rank The thread rank, used to know which cache to use
       * @param inputs The input tensor
       * @param targets The target tensor
       * @param queue The queue to use for the optimization
       */
      void operator()(size_t thread_rank, const math::clFTensor &inputs,
                      const math::clFTensor &targets, cl::CommandQueue queue) override {
        computeGradient(thread_rank, inputs, targets, queue);
      }

      /**
       * @brief Runs the optimizer, and return the error on the input
       * @param thread_rank The thread rank, used to know which cache to use
       * @param inputs The input tensor containing the input data
       * @param targets The target tensor containing the target associated with the input
       * @param queue The queue to use for the optimizationt
       * @return
       */
      math::clFTensor computeGradient(size_t thread_rank, const math::clFTensor &inputs,
                                      const math::clFTensor &targets, cl::CommandQueue queue);

      /**
       * @brief Allocate a number of caches for the optimizer. Can be used for multi-threading
       * optimally since each gpu will have its dedicated tensors
       * @param num_threads
       */
      void reserveCaches(size_t num_threads) override;

      WeightUpdateCache &getCache(size_t thread_rank) {
        if (thread_rank >= caches.size()) {
          throw std::runtime_error("Cache not allocated for thread " + std::to_string(thread_rank));
        }
        return *caches[thread_rank];
      }


    protected:
      std::vector<std::unique_ptr<WeightUpdateCache>> caches;
      MLPOptimizer *optimizer;

      void reduceAll(cl::CommandQueue &queue) override;
      void applyChanges(cl::CommandQueue &queue) override;
      void clearChanges(cl::CommandQueue &queue) override;
    };

    /**
     * @brief Builds a new optimizer for the given model. Note that the model should remain valid
     * for the lifetime of the optimizer.
     * @param model The model to optimize
     * @param optimization The optimization method to use. Takes ownership of the method
     */
    MLPOptimizer(MLPModel &model, std::unique_ptr<Optimization> optimization)
        : neural_network(&model.getPerceptron()), opti_meth(std::move(optimization)) {}

    /**
     * @brief Build a new optimizer for the given perceptron. Note that the perceptron should remain
     * valid for the lifetime of the optimizer.
     * @param perceptron The perceptron to optimize
     * @param optimization The optimization method to use. Takes ownership of the method
     */
    MLPOptimizer(MLPerceptron &perceptron, std::unique_ptr<Optimization> optimization)
        : neural_network(&perceptron), opti_meth(std::move(optimization)) {}

    /**
     * @brief Updates the optimizer and the optimization method. Should be called after each epoch
     */
    void update() override { opti_meth->update(); }

    /**
     * @brief Helper method, creates a new optimizer and optimization method and return the
     * optimizzer
     * @tparam optim The type of optimization method to use
     * @tparam Args The arguments to pass to the optimization method
     * @param model The model to optimize
     * @param args The args for the optimization method
     * @return A new optimizer
     */
    template<class optim, typename... Args>
    static std::unique_ptr<MLPOptimizer> make(MLPModel &model, Args &&...args) {
      return std::make_unique<MLPOptimizer>(
              model, std::make_unique<optim>(model.getPerceptron(), std::forward<Args>(args)...));
    }

    /**
     * @brief Return a delegate to run the optimizer
     * @return
     */
    std::unique_ptr<Operation> makeOperation() {
      // C++ doesn't support covariance with smart pointers
      // We use raw pointers here, before encapsulating them
      auto *ptr = makeOperationImpl();
      return std::unique_ptr<Operation>(ptr);
    }

    /**
     * @brief Perform a gradient descent on the model. Note that this algorithm doesn't wait for the
     * queue to finish
     * @param inputs An input tensor
     * @param targets A tensor containing the associated targets
     * @param cache A cache containing a copy of the weights and biases. Note that the cache will be
     * migrated to the GPU associated with the queue
     * @param queue The queue to use for the computation
     * @return The error on the input
     */
    math::clFTensor optimize(const math::clFTensor &inputs, const math::clFTensor &targets,
                             WeightUpdateCache &cache, cl::CommandQueue &queue);

    /**
     * @brief Create a cache that can be used with this object
     * @return
     */
    virtual std::unique_ptr<WeightUpdateCache> makeCache();

    /**
     * @brief Create a vector of caches that can be used with this object
     * @param ncache
     * @return
     */
    virtual std::vector<std::unique_ptr<WeightUpdateCache>> makeCaches(size_t ncache);

  private:
    /*
     * @brief Return a raw pointer to a newly allocated operation
     */
    Operation *makeOperationImpl() override;

    MLPerceptron *neural_network;
    std::unique_ptr<Optimization> opti_meth;
  };

  class MLPOptimizer::WeightUpdateCache {
  public:
    explicit WeightUpdateCache(MLPOptimizer &optimizer);
    WeightUpdateCache(std::vector<math::clFMatrix> weight_updates, size_t contributions);

    virtual ~WeightUpdateCache() = default;

    math::clFMatrix &operator[](size_t i) { return weight_updates[i]; }

    const math::clFMatrix &operator[](size_t i) const { return weight_updates[i]; }
    const std::vector<math::clFMatrix> &getWeightsCopy() const { return weight_copy; }
    const std::vector<math::clFMatrix> &getBiasesCopy() const { return biases_copy; }

    void add(size_t index, const math::clFMatrix &delta, size_t contribution_size,
             cl::CommandQueue &queue);
    void reduce(WeightUpdateCache &other, cl::CommandQueue &queue);

    void apply(cl::CommandQueue &queue);

    void increaseContribution(size_t contrib) { contribution += contrib; }

    void synchronizeWeights(cl::CommandQueue &queue);
    void acquireBuffer(cl::CommandQueue &queue);

    void clear(cl::CommandQueue &queue);

  protected:
    MLPerceptron *perceptron;
    size_t contribution;
    std::vector<math::clFMatrix> weight_updates;
    std::vector<math::clFMatrix> weight_copy;
    std::vector<math::clFMatrix> biases_copy;

  private:
    Optimization *optimization;
  };

}   // namespace nnet