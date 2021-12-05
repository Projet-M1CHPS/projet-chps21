#pragma once

#include "Image.hpp"
#include "Transform.hpp"
#include "inputSet.hpp"
#include "trainingCollection.hpp"
#include <filesystem>
#include <set>


namespace control {

  class SetLoader {
  public:
    [[nodiscard]] virtual InputSet load(std::filesystem::path const &input_path, bool verbose,
                                        std::ostream *out) = 0;

  private:
  };

  class ImageSetLoader : public SetLoader {
  public:
    ImageSetLoader() {}

  private:
  };

  template<class TrainingSet>
  class TSLoader {
  public:
    [[nodiscard]] virtual std::shared_ptr<TrainingSet> load(std::filesystem::path const &input_path,
                                                            bool verbose, std::ostream *out) = 0;
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
}   // namespace control
