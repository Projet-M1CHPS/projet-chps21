#include "imageInputSet.hpp"

namespace control {

  ImageInputSet ImageInputSetLoader::load(const std::filesystem::path &path) {
    if (!std::filesystem::exists(path)) {
      throw std::runtime_error("ImageInputSetLoader::load: path does not exist");
    }
    if (is_directory(path)) return loadDirectory(path, nullptr);
    else if (is_regular_file(path))
      return loadFile(path, nullptr);
    else
      throw std::runtime_error("ImageInputSetLoader::load: path is neither a directory nor a file");
  }

  std::pair<ImageInputSet, std::vector<std::filesystem::path>>
  ImageInputSetLoader::loadWithPaths(const std::filesystem::path &path) {
    if (!std::filesystem::exists(path)) {
      throw std::runtime_error("ImageInputSetLoader::loadWithPaths: path does not exist");
    }
    std::vector<std::filesystem::path> paths;
    if (is_directory(path)) return {loadDirectory(path, &paths), paths};
    else if (is_regular_file(path))
      return {loadFile(path, &paths), paths};
    else
      throw std::runtime_error(
              "ImageInputSetLoader::loadWithPaths: path is neither a directory nor a file");
  }

  ImageInputSet ImageInputSetLoader::loadFile(const std::filesystem::path &path,
                                              std::vector<std::filesystem::path> *paths) {
    ImageInputSet res;

    res.append(image::ImageSerializer::load(path));
    if (paths) paths->push_back(path);
    return res;
  }

  ImageInputSet ImageInputSetLoader::loadDirectory(const std::filesystem::path &path,
                                                   std::vector<std::filesystem::path> *paths) {
    ImageInputSet res;
    loadDirectory(path, res, paths);
    return res;
  }

  void ImageInputSetLoader::loadDirectory(const std::filesystem::path &path, ImageInputSet &set,
                                          std::vector<std::filesystem::path> *paths) {
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
      // Recurse into subdirectories
      if (is_directory(entry.path())) {
        loadDirectory(entry.path(), set, paths);
      } else if (is_regular_file(entry.path())) {
        set.append(image::ImageSerializer::load(entry.path()));
        paths->push_back(entry.path());
      }
    }
  }

}   // namespace control
