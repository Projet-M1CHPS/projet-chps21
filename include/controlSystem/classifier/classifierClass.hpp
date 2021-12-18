#pragma once
#include <iostream>
#include <vector>

/** Represents a of objects, to be used in a classifier.
 * A class is uniquely identified by an id, and contains a name (that may not be unique)
 */
class ClassLabel {
  friend std::ostream &operator<<(std::ostream &os, const ClassLabel &label);

public:
  ClassLabel(size_t id, std::string name) : id(id), name(std::move(name)) {}
  [[nodiscard]] size_t getId() const { return id; }
  [[nodiscard]] std::string const &getName() const { return name; }

  void setId(size_t id) { this->id = id; }
  void setName(std::string name) { this->name = std::move(name); }

  /** Returns a label corresponding to the unknown class
   *
   * This means that the id = 0 and the name = "unknown"
   *
   * Marked noexcept to allow the creation of the static object "unknown"
   *
   * @return
   */
  static ClassLabel const &getUnknown() noexcept {
    static ClassLabel unknown(0, "Unknown");
    return unknown;
  }

  /** A global object corresponding to the unknown class
   *
   */
  inline static ClassLabel const &unknown = getUnknown();

  bool operator==(ClassLabel const &other) const { return id == other.id && name == other.name; }
  bool operator!=(ClassLabel const &other) const { return !(*this == other); }

  bool operator>(ClassLabel const &other) const { return id > other.id; }
  bool operator<(ClassLabel const &other) const { return id < other.id; }


private:
  size_t id;
  std::string name;
};

/** Represents a list of classes that can be outputted by a classifier
 *
 */
class ClassifierClassLabelList {
public:
  size_t size() const { return labels.size(); }

  bool empty() const { return labels.empty(); }

  void append(ClassLabel label);

  using iterator = std::vector<ClassLabel>::iterator;
  using const_iterator = std::vector<ClassLabel>::const_iterator;

  iterator begin() { return labels.begin(); }
  iterator end() { return labels.end(); }

  const_iterator begin() const { return labels.begin(); }
  const_iterator end() const { return labels.end(); }


private:
  std::vector<ClassLabel> labels;
};