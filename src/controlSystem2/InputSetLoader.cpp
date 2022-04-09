#include "InputSetLoader.hpp"
#include <execution>
#include <filesystem>
#include <stack>

namespace fs = std::filesystem;
namespace tr = image::transform;

namespace control {
  namespace {
    /**
     * @brief Helper class for image loading and pipelining
     */
    class TransformationPipeline {
    public:
      /**
       * @brief Build a pipeline from a list of transformations
       * @param transformations
       */
      TransformationPipeline(std::initializer_list<tr::TransformEngine> transformations)
          : engines(transformations) {}

      /**
       * @brief Apply the pipeline to an image loaded from disk
       * @param image_path Path to the image to load
       * @return The transformed image
       */
      image::GrayscaleImage loadAndTransform(const fs::path &image_path) const;

    private:
      /**
       * @brief The list of transformations to apply when loading an image
       */
      std::vector<tr::TransformEngine> engines;
    };

    image::GrayscaleImage
    TransformationPipeline::loadAndTransform(const fs::path &image_path) const {
      image::GrayscaleImage image = image::ImageSerializer::load(image_path);
      for (auto &engine : engines) { engine.apply(image); }
      return image;
    }

    // Store metadata for an input sample
    struct InputFileMetadata {
      // The path of the sample on disk (path to an image)
      fs::path path;
      // The unique id of the sample
      // (Ids are unique across all samples in this InputSet)
      // Ids can be reused over multiple InputSet
      size_t input_id;
      // The id of the class
      long class_id;
    };

    // Take a list of input files, create a Tensor and fill it up with images loaded from the files.
    // Images are normalized and transformed, and stored inside the Tensor (Using the default
    // device) The tensor is then appended at the end of the input set.
    void loadTensor(const std::vector<InputFileMetadata> &files, size_t index, size_t tensor_size,
                    TransformationPipeline &pipeline, InputSet &res) {
      // We may be left with a small number of images, so we need to resize the tensor
      // This can happen if the number of images is not a multiple of the tensor size
      size_t count = std::min(tensor_size, files.size() - index);
      math::clFTensor tensor(res.getInputWidth(), res.getInputHeight(), count);

      // Create an out-of-order queue to transform the images in parallel
      cl::CommandQueue queue = cl::CommandQueue(utils::cl_wrapper.getContext(),
                                                utils::cl_wrapper.getDefaultDevice());

      // Fetch the normalizing kernel
      // This kernel convert a char array to a float array, and scale every element by a given
      // factor
      cl::Kernel kernel = utils::cl_wrapper.getKernels().getKernel("NormalizeCharToFloat.cl",
                                                                   "normalizeCharToFloat");


      std::vector<size_t> ids;
      std::vector<long> classes;

      for (size_t i = 0; i < count; ++i) {
        auto &file = files[index + i];
        // Load and transform the input image
        auto image = pipeline.loadAndTransform(file.path);

        // Copy the image to the Device
        // We need to load the image to the cl device before applying the kernel
        cl::Buffer img_buffer(CL_MEM_READ_WRITE, image.getSize());
        queue.enqueueWriteBuffer(img_buffer, CL_TRUE, 0, image.getSize(), image.getData());

        kernel.setArg(0, img_buffer);
        auto matrix = tensor.getMatrix(i);
        kernel.setArg(1, matrix.getBuffer());
        kernel.setArg(2, matrix.getOffset());

        // We also normalize the matrix by a given factor
        kernel.setArg(3, 255.0f);

        // Convert the image to float and normalize it by 255
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image.getSize()),
                                   cl::NullRange);
        ids.push_back(file.input_id);
        classes.push_back(file.class_id);
      }
      // Wait until all the images are loaded
      queue.finish();

      // Append the tensor to the input set
      res.append(std::move(tensor), ids, classes);
    }

    // Take a list of samples metadata, and load them
    void loadFiles(InputSet &res, size_t tensor_size, const tr::TransformEngine &pre_engine,
                   const tr::TransformEngine &post_engine, std::vector<InputFileMetadata> &files,
                   bool shuffle_samples) {
      // Shuffle the samples if needed
      if (shuffle_samples)
        std::shuffle(files.begin(), files.end(), std::mt19937(std::random_device{}()));

      // Create the transformation pipeline
      tr::TransformEngine resize_engine;
      resize_engine.addTransformation(
              std::make_shared<tr::Resize>(res.getInputWidth(), res.getInputHeight()));
      TransformationPipeline pipeline({resize_engine, pre_engine, post_engine});


      for (size_t index = 0; index < files.size();
           index += std::min(files.size() - index, tensor_size)) {
        // Let async handle the threading policy
        loadTensor(files, index, tensor_size, pipeline, res);
      }
    }
  }   // namespace

  InputSet InputSetLoader::loadWithClasses(const std::filesystem::path &path,
                                           bool shuffle_samples) const {
    if (not fs::exists(path))
      throw std::runtime_error("InputSetLoader::loadWithClasses: The input path does not exist");
    InputSet res(input_width, input_height);

    std::vector<InputFileMetadata> files;
    std::vector<std::string> classes;

    // Load the following layout :
    // <root_dir>
    //   <class_1> (class_id: 0)
    //     <image_1> (class_id: 0, id: 0)
    //     <image_2> (class_id: 0, id: 1)
    //   <class_2>
    //     <image_1> (class_id: 1, id: 2)
    //     <image_2> (class_id: 1, id: 3)
    //   ...
    for (const auto &entry : fs::directory_iterator(path)) {
      if (not entry.is_directory()) continue;

      // Add a new class
      classes.push_back(entry.path().filename().string());

      for (const auto &file : fs::directory_iterator(entry.path())) {
        if (not file.is_regular_file()) continue;

        // TODO fix me
        files.push_back({file.path(), files.size(), static_cast<long>(classes.size() - 1)});
      }
    }

    tscl::logger("Loading " + std::to_string(files.size()) + " samples with classes");
    // Load all the samples
    loadFiles(res, tensor_size, preprocess_engine, postprocess_engine, files, shuffle_samples);
    // Update the classes with the new classes
    res.updateClasses(classes);
    return res;
  }

  InputSet InputSetLoader::loadWithoutClasses(const std::filesystem::path &path,
                                              bool shuffle_samples) const {
    InputSet res(input_width, input_height);

    std::vector<InputFileMetadata> files;

    std::stack<std::filesystem::path> dirs;
    dirs.push(path);

    // Load the following layout :
    // <root_dir>
    //   <image_1> (class_id: -1, id: 0)
    //   <image_2> (class_id: -1, id: 1)
    //   <subdir_1>
    //     <image_1> (class_id: -1, id: 2)
    //     <image_2> (class_id: -1, id: 3)
    //   ...
    while (not dirs.empty()) {
      // Load the next directory in the stack
      auto curr_dir = dirs.top();
      dirs.pop();

      for (const auto &entry : fs::directory_iterator(curr_dir)) {
        if (entry.is_directory()) {
          // Defer the subdirectory
          dirs.push(entry.path());
        } else if (entry.is_regular_file()) {
          // Store metadata about this sample
          files.push_back({entry.path(), files.size(), -1});
        }
      }
    }

    tscl::logger("Loading " + std::to_string(files.size()) + " samples without classes");

    // Load all the samples
    loadFiles(res, tensor_size, preprocess_engine, postprocess_engine, files, shuffle_samples);
    return res;
  }

  InputSet InputSetLoader::load(const std::filesystem::path &path, bool load_classes,
                                bool shuffle_samples) const {
    tscl::logger("Loading input set from " + path.string(), tscl::Log::Information);
    if (load_classes) return loadWithClasses(path, shuffle_samples);
    else
      return loadWithoutClasses(path, shuffle_samples);
  }
}   // namespace control