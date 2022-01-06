#pragma once
#include "Image.hpp"
#include "Transform.hpp"
#include "classifierInputSet.hpp"

namespace control::classifier {

  /** A collection used for training classifier model
   *
   * Stores two set of inputs, one for training and one for evaluation
   * Every input (in both sets) is uniquely identified by its id, which can be used for metadata
   * retrieving
   *
   */
  class CTCollection {
    friend std::ostream &operator<<(std::ostream &os, CTCollection const &set);

  public:
    /** Create a collection with a given label list
     *
     *
     * @param classes
     */
    explicit CTCollection(std::shared_ptr<CClassLabelSet> classes);

    /** Collection can be really huge, so we delete the copy operators for safety
     *
     * @param other
     */
    // FIXME Add a copy method
    CTCollection(CTCollection const &other) = delete;

    CTCollection(CTCollection &&other) = default;
    CTCollection &operator=(CTCollection &&other) = default;

    /** Return the training set
     *
     * @param set
     */
    [[nodiscard]] ClassifierTrainingSet &getTrainingSet() { return training_set; }
    [[nodiscard]] ClassifierTrainingSet const &getTrainingSet() const { return training_set; }

    /** Returns the evaluation set
     *
     * @return
     */
    [[nodiscard]] ClassifierTrainingSet &getEvalSet() { return eval_set; };
    [[nodiscard]] ClassifierTrainingSet const &getEvalSet() const { return eval_set; };

    /** Returns the number of classes in this collection
     *
     * @return
     */
    [[nodiscard]] size_t getClassCount() const { return class_list->size(); }

    /** Return the classes used in this collection
     *
     * @return
     */
    [[nodiscard]] CClassLabelSet const &getClasses() const { return *class_list; }

    /** Randomly shuffle the training set
     *
     * Training should be done on a shuffled set for a uniform learning that is not biased by the
     * inputs order
     *
     * @param seed
     */
    void shuffleTrainingSet(size_t seed) { training_set.shuffle(seed); }

    /* Randomly shuffle the eval set
     *
     */
    void shuffleEvalSet(size_t seed) { eval_set.shuffle(seed); }

    /** Randomly shuffles both sets with a seed
     * The same seed is used for both shuffling
     *
     * @param seed
     */
    void shuffleSets(size_t seed) {
      shuffleTrainingSet(seed);
      shuffleEvalSet(seed);
    }

    /** Returns true if both sets are empty
     *
     * @return
     */
    [[nodiscard]] bool empty() const { return training_set.empty() && eval_set.empty(); }

    /** Returns the sum of the sizes of both sets
     *
     * @return
     */
    [[nodiscard]] size_t size() const { return training_set.size() + eval_set.size(); }

    /** Unload both sets, freeing any used memory
     *
     */
    void unload() {
      training_set.clear();
      eval_set.clear();
      class_list = nullptr;
    }

  private:
    ClassifierTrainingSet training_set, eval_set;
    std::shared_ptr<CClassLabelSet> class_list;
  };


  /** Interface for a classifier training collection loader
   *
   */
  class CTCLoader {
  public:
    virtual ~CTCLoader() = default;

    /** Set the classes that will be used by the loader
     * This function takes a shared ptr to the class list
     * since the loader has to build one himself if none is provided
     *
     * @param list
     */
    void setClasses(std::shared_ptr<CClassLabelSet> list) { classes = std::move(list); }

    [[nodiscard]] CClassLabelSet &getClasses() { return *classes; }
    [[nodiscard]] CClassLabelSet &getClasses() const { return *classes; }

  protected:
    std::shared_ptr<CClassLabelSet> classes;
  };

  /** Classifier collection loader for image inputs
   *
   */
  class CITCLoader : public CTCLoader {
  public:
    /** Creates a loader that will rescale the image to a given size before inserting them in the
     * collection
     *
     * @param width
     * @param height
     */
    CITCLoader(const size_t width, const size_t height)
        : target_width(width), target_height(height) {}

    /** Loads a collection from a directory
     *
     * @param input_path a path to a directory containing the evaluation and training set
     * @return
     */
    [[nodiscard]] std::unique_ptr<CTCollection> load(const std::filesystem::path &input_path);

    /** Returns the transformation engine that gets applied before the rescaling
     *
     * @return
     */
    [[nodiscard]] image::transform::TransformEngine &getPreProcessEngine() { return pre_process; }
    [[nodiscard]] image::transform::TransformEngine const &getPreProcessEngine() const {
      return pre_process;
    }

    /** Returns the transformations engine that gets applied after the rescaling
     *
     * Post process transformations should be aimed at enhancing the contrast of the image
     * to counter the loss of the rescaling
     *
     * @return
     */
    [[nodiscard]] image::transform::TransformEngine &getPostProcessEngine() { return post_process; }
    [[nodiscard]] image::transform::TransformEngine const &getPostProcessEngine() const {
      return post_process;
    }

  private:
    /** Only called if no classes are provided by the user
     * Automatically fetch the class from the sub-directories name
     * @param input_path
     */
    void loadClasses(std::filesystem::path const &input_path);
    void loadEvalSet(CTCollection &res, const std::filesystem::path &input_path);
    void loadTrainingSet(CTCollection &res, std::filesystem::path const &input_path);

    void loadSet(ClassifierTrainingSet &res, std::filesystem::path const &input_path);

    image::transform::TransformEngine pre_process, post_process;

    /** Rescaling size
     *
     */
    size_t target_width, target_height;
  };
}   // namespace control::classifier