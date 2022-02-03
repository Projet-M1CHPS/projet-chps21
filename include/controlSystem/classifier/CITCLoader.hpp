#pragma once
#include "Image.hpp"
#include "Transform.hpp"
#include "clWrapper.hpp"
#include <CL/opencl.hpp>
#include <CTCollection.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <tscl.hpp>

namespace control::classifier {

  /** @brief Classifier collection loader for image inputs
   *
   */
  class CITCLoader {
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
    [[nodiscard]] std::unique_ptr<CTCollection> load(const std::filesystem::path &input_path,
                                                     utils::clWrapper &wrapper);

    /** Returns the transformation engine that gets applied before the rescaling
     *
     * @return
     */
    [[nodiscard]] image::transform::TransformEngine &getPreProcessEngine() { return pre_process; }
    [[nodiscard]] image::transform::TransformEngine const &getPreProcessEngine() const {
      return pre_process;
    }

    /** Set the classes that will be used by the loader
     * This function takes a shared ptr to the class list
     * since the loader has to build one himself if none is provided
     *
     * @param list
     */
    void setClasses(std::shared_ptr<CClassLabelSet> list) { classes = std::move(list); }

    [[nodiscard]] CClassLabelSet &getClasses() { return *classes; }
    [[nodiscard]] CClassLabelSet &getClasses() const { return *classes; }

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

    void loadSet(CTrainingSet &res, std::filesystem::path const &input_path,
                 utils::clWrapper &wrapper);

    image::transform::TransformEngine pre_process, post_process;

    /** Rescaling size
     *
     */
    size_t target_width, target_height;
    std::shared_ptr<CClassLabelSet> classes;
  };
}   // namespace control::classifier