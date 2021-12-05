#pragma once
#include "classifierInputSet.hpp"
#include "controlSystem/inputSetLoader.hpp"

namespace control::classifier {

  class ClassifierTrainingCollection {
    friend std::ostream &operator<<(std::ostream &os, ClassifierTrainingCollection const &set);

  public:
    ClassifierTrainingCollection();
    ClassifierTrainingCollection(std::shared_ptr<std::set<ClassLabel>> classes);

    ClassifierTrainingCollection(ClassifierTrainingCollection const &other) = delete;

    ClassifierTrainingCollection(ClassifierTrainingCollection &&other) = default;
    ClassifierTrainingCollection &operator=(ClassifierTrainingCollection &&other) = default;

    [[nodiscard]] ClassifierInputSet &getTrainingSet() { return training_set; }
    [[nodiscard]] ClassifierInputSet const &getTrainingSet() const { return training_set; }

    [[nodiscard]] ClassifierInputSet &getEvalSet() { return eval_set; };
    [[nodiscard]] ClassifierInputSet const &getEvalSet() const { return eval_set; };

    [[nodiscard]] size_t categoryCount() const { return class_labels->size(); }

    [[nodiscard]] std::set<ClassLabel> &getLabels() { return *class_labels; }
    [[nodiscard]] std::set<ClassLabel> const &getLabels() const { return *class_labels; }

    template<typename iterator>
    void setCategories(iterator begin, iterator end) {
      class_labels->clear();
      class_labels->insert(class_labels->begin(), begin, end);
    }

    void shuffleTrainingSet(size_t seed);
    void shuffleEvalSet(size_t seed);
    void shuffleSets(size_t seed);

    [[nodiscard]] bool empty() const { return training_set.empty() && eval_set.empty(); }
    [[nodiscard]] size_t size() const { return training_set.size() + eval_set.size(); }

    void unload();

  private:
    ClassifierInputSet training_set, eval_set;
    std::shared_ptr<std::set<ClassLabel>> class_labels;
  };


  class CTCLoader : public TSLoader<ClassifierTrainingCollection> {
  public:
    template<typename iterator>
    void setClasses(iterator begin, iterator end) {
      if (not classes) classes = std::make_shared<std::set<ClassLabel>>();
      classes->clear();
      for (auto it = begin; it != end; ++it) { classes->insert(*it); }
    }

    [[nodiscard]] std::shared_ptr<std::set<ClassLabel>> getClasses() { return classes; }
    [[nodiscard]] std::shared_ptr<std::set<ClassLabel>> getClasses() const { return classes; }

  protected:
    std::shared_ptr<std::set<ClassLabel>> classes;
  };

  class CITCLoader : public CTCLoader {
  public:
    CITCLoader(size_t width, size_t height) : target_width(width), target_height(height) {}

    [[nodiscard]] std::shared_ptr<ClassifierTrainingCollection>
    load(std::filesystem::path const &input_path, bool verbose, std::ostream *out) override;

    [[nodiscard]] image::transform::TransformEngine &getPreProcessEngine() { return pre_process; }
    [[nodiscard]] image::transform::TransformEngine const &getPreProcessEngine() const {
      return pre_process;
    }

    [[nodiscard]] image::transform::TransformEngine &getPostProcessEngine() { return post_process; }
    [[nodiscard]] image::transform::TransformEngine const &getPostProcessEngine() const {
      return post_process;
    }

  private:
    void loadClasses(std::filesystem::path const &input_path);
    void loadEvalSet(ClassifierTrainingCollection &res, const std::filesystem::path &input_path,
                     bool verbose, std::ostream *out);
    void loadTrainingSet(ClassifierTrainingCollection &res, std::filesystem::path const &input_path,
                         bool verbose, std::ostream *out);

    void loadSet(ClassifierInputSet &res, std::filesystem::path const &input_path);

    image::transform::TransformEngine pre_process;
    image::transform::TransformEngine post_process;
    size_t target_width, target_height;
  };
}   // namespace control::classifier