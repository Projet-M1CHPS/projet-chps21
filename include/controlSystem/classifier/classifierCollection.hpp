#pragma once
#include "classifierInputSet.hpp"
#include "controlSystem/inputSetLoader.hpp"

namespace control::classifier {

  /** A collection of a training set and an eval set used for training a classifier
   *
   * Store both set along a list of labels for each class
   *
   */
  class ClassifierTrainingCollection {
    friend std::ostream &operator<<(std::ostream &os, ClassifierTrainingCollection const &set);

  public:
    /** Create a collection with an empty label list
     *
     */
    ClassifierTrainingCollection();

    /** Create a collection with a given label list
     *
     * @param classes
     */
    explicit ClassifierTrainingCollection(std::shared_ptr<std::vector<ClassLabel>> classes);

    ClassifierTrainingCollection(ClassifierTrainingCollection const &other) = delete;

    ClassifierTrainingCollection(ClassifierTrainingCollection &&other) = default;
    ClassifierTrainingCollection &operator=(ClassifierTrainingCollection &&other) = default;

    /** Return the training set
     *
     * @param set
     */
    [[nodiscard]] ClassifierInputSet &getTrainingSet() { return training_set; }
    [[nodiscard]] ClassifierInputSet const &getTrainingSet() const { return training_set; }

    /** Returns the eval set
     *
     * @return
     */
    [[nodiscard]] ClassifierInputSet &getEvalSet() { return eval_set; };
    [[nodiscard]] ClassifierInputSet const &getEvalSet() const { return eval_set; };

    /** Returns the number of classes in this collection
     *
     * @return
     */
    [[nodiscard]] size_t classCount() const { return class_labels->size(); }

    /** Return the classes used in this collection
     *
     * @return
     */
    [[nodiscard]] std::vector<ClassLabel> const &getClasses() const { return *class_labels; }

    /** Set the classes used in this collection
     *
     * Be warned that the eval and training sets will be cleared as they both become invalid
     *
     * @tparam iterator
     * @param begin
     * @param end
     */
    template<typename iterator>
    void setClasses(iterator begin, iterator end) {
      class_labels->clear();
      training_set.unload();
      eval_set.unload();
      class_labels->insert(class_labels->begin(), begin, end);
    }

    /** Randomly shuffle the training set
     *
     * Training should be done on a shuffled set for a uniform learning that is not biased by the
     * order of the data and will not only focus on the last class
     *
     * @param seed
     */
    void shuffleTrainingSet(size_t seed) { training_set.shuffle(seed); }

    /* Randomly shuffle the eval set
     *
     */
    void shuffleEvalSet(size_t seed) { eval_set.shuffle(seed); }

    /** Randomly shuffles both sets with a seed
     * The seed is used for both shuffling
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

    /** Unload both sets
     *
     */
    void unload() {
      training_set.unload();
      eval_set.unload();
    }

  private:
    ClassifierInputSet training_set, eval_set;
    std::shared_ptr<std::vector<ClassLabel>> class_labels;
  };


  /** Abstract base for classifier training collections loader
   *
   */
  class CTCLoader : public TrainingSetLoader<ClassifierTrainingCollection> {
  public:
    virtual ~CTCLoader() = default;
    template<typename iterator>
    void setClasses(iterator begin, iterator end) {
      if (not classes) classes = std::make_shared<std::vector<ClassLabel>>();
      classes->clear();
      classes->insert(classes->begin(), begin, end);

      std::sort(classes->begin(), classes->end());
    }

    [[nodiscard]] std::shared_ptr<std::vector<ClassLabel>> getClasses() { return classes; }
    [[nodiscard]] std::shared_ptr<std::vector<ClassLabel> const> getClasses() const {
      return classes;
    }

  protected:
    std::shared_ptr<std::vector<ClassLabel>> classes;
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
    CITCLoader(size_t width, size_t height) : target_width(width), target_height(height) {}

    /** Loads the collection pointed to by the input path
     *
     * @param input_path
     * @param verbose
     * @param out
     * @return
     */
    [[nodiscard]] std::shared_ptr<ClassifierTrainingCollection>
    load(std::filesystem::path const &input_path) override;

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
    void loadClasses(std::filesystem::path const &input_path);
    void loadEvalSet(ClassifierTrainingCollection &res, const std::filesystem::path &input_path);
    void loadTrainingSet(ClassifierTrainingCollection &res,
                         std::filesystem::path const &input_path);

    void loadSet(ClassifierInputSet &res, std::filesystem::path const &input_path);

    image::transform::TransformEngine pre_process;
    image::transform::TransformEngine post_process;
    size_t target_width, target_height;
  };
}   // namespace control::classifier