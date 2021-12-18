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
     * This is used for
     *
     * @param classes
     */
    explicit CTCollection(std::shared_ptr<ClassifierClassLabelList> classes);

    /** Collection can be really huge, so we delete the copy operators for safety
     * FIXME: add a copy method
     *
     * @param other
     */
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
    [[nodiscard]] ClassifierClassLabelList const &getClasses() const { return *class_list; }

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

    /** Unload both sets, freeing any used memory
     *
     */
    void unload() {
      training_set.unload();
      eval_set.unload();
      class_list = nullptr;
    }

  private:
    ClassifierTrainingSet training_set, eval_set;
    std::shared_ptr<ClassifierClassLabelList> class_list;
  };


  /** Interface for a classifier training collection loader
   *
   */
  class CTCLoader {
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
    [[nodiscard]] std::shared_ptr<CTCollection> load(std::filesystem::path const &input_path);

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
    void loadEvalSet(CTCollection &res, const std::filesystem::path &input_path);
    void loadTrainingSet(CTCollection &res, std::filesystem::path const &input_path);

    void loadSet(ClassifierTrainingSet &res, std::filesystem::path const &input_path);

    image::transform::TransformEngine pre_process;
    image::transform::TransformEngine post_process;
    size_t target_width, target_height;
  };
}   // namespace control::classifier