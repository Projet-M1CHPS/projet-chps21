#pragma once
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>

namespace utils {

  /**
   * @brief Exit the program with an error message. Should only be used where a
   * throw is not enough
   *
   * @param msg
   */
  [[noreturn]] void error(const std::string &msg) noexcept;

  /**
   * @brief Exit the program with an error message. Should only be used where a
   * throw is not enough
   *
   * @param msg
   */
  [[noreturn]] void error(const char *msg) noexcept;

  template<typename ArrayType>
  struct aligned_deleter {
    void operator()(ArrayType *array) {
      if (array) {
        free(array);
        array = nullptr;
      }
    }
  };

  template<typename T>
  using unique_aligned_ptr = std::unique_ptr<T[], aligned_deleter<T>>;

  template<typename T>
  unique_aligned_ptr<T> make_aligned_unique(size_t align, size_t size) {
    return unique_aligned_ptr<T>(static_cast<T *>(aligned_alloc(align, size * sizeof(T))),
                                 aligned_deleter<T>());
  }

  template<typename T>
  using shared_aligned_ptr = std::shared_ptr<T[]>;

  template<typename T>
  shared_aligned_ptr<T> make_aligned_shared(size_t align, size_t size) {
    return shared_aligned_ptr<T>(static_cast<T *>(aligned_alloc(align, size * sizeof(T))),
                                 aligned_deleter<T>());
  }

  std::string timestampAsStr();

  // Generic IO exception
  class IOException : public std::runtime_error {
  public:
    explicit IOException(const std::string &msg) noexcept;
    explicit IOException(const char *msg) noexcept;
  };

}   // namespace utils