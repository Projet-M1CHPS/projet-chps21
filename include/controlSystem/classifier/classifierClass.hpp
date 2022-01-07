#pragma once
#include <iostream>
#include <map>
#include <vector>

namespace control::classifier {

  /** @brief Class unique identifier
   *
   *  Classes are uniquely identified by an id, but their also associated a name for
   *  printing purposes
   */
  class ClassLabel {
    friend std::ostream &operator<<(std::ostream &os, const ClassLabel &label);

  public:
    // This is needed for storing classes in a map
    ClassLabel() { *this = getUnknown(); }
    ClassLabel(size_t id, std::string name) : id(id), name(std::move(name)) {}

    /**
     * @return the uniaue id of the class
     */
    [[nodiscard]] size_t getId() const { return id; }
    [[nodiscard]] std::string const &getName() const { return name; }

    /** The id should be unique for a given class set
     *
     * @param id The new id of the class
     */
    void setId(size_t id) { this->id = id; }

    /**
     *
     * @param The class name. Not necessarily unique
     */
    void setName(std::string name) { this->name = std::move(name); }

    /** Returns a label corresponding to the unknown class
     *
     * This means that the id = 0 and name = "unknown"
     * Marked noexcept to allow the creation of the static object "unknown"
     *
     * @return a special label used to identify the unknown class
     */
    static ClassLabel const &getUnknown() noexcept {
      static ClassLabel unknown(0, "Unknown");
      return unknown;
    }

    /** A global object corresponding to the unknown class
     *
     */
    inline static ClassLabel const &unknown = getUnknown();

    /**
     *
     * @param other
     * @return True if both the id and the name are the same, false otherwise
     */
    bool operator==(ClassLabel const &other) const { return id == other.id && name == other.name; }
    bool operator!=(ClassLabel const &other) const { return !(*this == other); }

    /**
     *
     * @param other
     * @return True if the id  is superior to the other id, false otherwise
     */
    bool operator>(ClassLabel const &other) const { return id > other.id; }
    bool operator<(ClassLabel const &other) const { return id < other.id; }


  private:
    size_t id;
    std::string name;
  };

  /** @brief Set of ClassLabel to be used in a classifier algorithm
   *
   */
  class CClassLabelSet {
    friend std::ostream &operator<<(std::ostream &os, const CClassLabelSet &label);

  public:
    /**
     * @return the size of the set
     */
    [[nodiscard]] size_t size() const { return labels.size(); }

    /**
     *
     * @return true if the set is empty, false otherwise
     */
    [[nodiscard]] bool empty() const { return labels.empty(); }

    /** Append a label to the set, and checks for id uniqueness
     *
     * @param label
     */
    void append(ClassLabel label);

    /** Return the label corresponding to the given id
     *
     * Throws on error
     * @param id the id of the label to return
     * @return the label corresponding to the given id
     */
    ClassLabel &operator[](size_t id) {
      if (not labels.contains(id)) {
        throw std::runtime_error("ClassLabelSet::operator[]: id not found");
      }
      return labels[id];
    }

    ClassLabel const &operator[](size_t id) const {
      if (not labels.contains(id)) {
        throw std::runtime_error("ClassLabelSet::operator[]: id not found");
      }
      return labels.find(id)->second;
    }


    using iterator = std::map<size_t, ClassLabel>::iterator;
    using const_iterator = std::map<size_t, ClassLabel>::const_iterator;

    iterator begin() { return labels.begin(); }
    iterator end() { return labels.end(); }

    [[nodiscard]] const_iterator begin() const { return labels.begin(); }
    [[nodiscard]] const_iterator end() const { return labels.end(); }


  private:
    std::map<size_t, ClassLabel> labels;
  };
}   // namespace control::classifier