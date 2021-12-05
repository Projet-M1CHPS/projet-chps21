
#pragma once
#include "Matrix.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace control::classifier {

  class CTrackerState {
  public:
    template<typename iterator>
    CTrackerState(std::filesystem::path const &output_path, iterator const class_begin,
                  iterator const class_end) {
      size_t size = std::distance(class_begin, class_end);

      auto output_dir = output_path / "class_data";
      std::filesystem::create_directories(output_dir);

      avg_prec_output = std::make_shared<std::ofstream>(output_dir / "avg_prec.dat");
      avg_recall_output = std::make_shared<std::ofstream>(output_dir / "avg_recall.dat");
      avg_f1_output = std::make_shared<std::ofstream>(output_dir / "avg_f1.dat");

      prec_outputs = std::make_shared<std::vector<std::ofstream>>();
      recall_outputs = std::make_shared<std::vector<std::ofstream>>();
      f1_outputs = std::make_shared<std::vector<std::ofstream>>();

      for (auto it = class_begin; it != class_end; it++) {
        prec_outputs->emplace_back(output_dir / (*it + "_prec.dat"));
        recall_outputs->emplace_back(output_dir / (*it + "_recall.dat"));
        f1_outputs->emplace_back(output_dir / (*it + "_f1.dat"));
      }
    }

    std::ostream &getAvgPrecOutput() { return *avg_prec_output; }
    std::ostream &getAvgRecallOutput() { return *avg_recall_output; }
    std::ostream &getAvgF1Output() { return *avg_f1_output; }

    std::ostream &getPrecOutput(size_t index) {
      if (not prec_outputs or index > prec_outputs->size())
        throw std::invalid_argument("CTrackerState: Invalid stream index");
      return prec_outputs->at(index);
    }

    std::ostream &getRecallOutput(size_t index) {
      if (not recall_outputs or index > recall_outputs->size())
        throw std::invalid_argument("CTrackerState: Invalid stream index");
      return recall_outputs->at(index);
    }

    std::ostream &getF1Output(size_t index) {
      if (not f1_outputs or index > f1_outputs->size())
        throw std::invalid_argument("CTrackerState: Invalid stream index");
      return f1_outputs->at(index);
    }

  private:
    std::shared_ptr<std::ofstream> avg_prec_output;
    std::shared_ptr<std::ofstream> avg_recall_output;
    std::shared_ptr<std::ofstream> avg_f1_output;

    std::shared_ptr<std::vector<std::ofstream>> prec_outputs;
    std::shared_ptr<std::vector<std::ofstream>> recall_outputs;
    std::shared_ptr<std::vector<std::ofstream>> f1_outputs;
  };

  class CTracker;

  class CStats {
    friend CTracker;
    friend std::ostream &operator<<(std::ostream &os, CStats const &stat);

  public:
    CStats() = delete;
    void classification_report(std::ostream &os, const std::vector<std::string> &classes);
    void dumpToFiles();

  private:
    CStats(size_t epoch, math::Matrix<size_t> const &confusion,
           std::shared_ptr<CTrackerState> state);

    size_t epoch;
    std::shared_ptr<CTrackerState> state;
    math::DoubleMatrix stats;
    double avg_precision, avg_recall, avg_f1score;
  };

  std::ostream &operator<<(std::ostream &os, CStats const &stat);

  class CTracker {
  public:
    template<typename iterator>
    CTracker(std::filesystem::path const &output_path, iterator const class_begin,
             iterator const class_end)
        : epoch(0), state(std::make_shared<CTrackerState>(output_path, class_begin, class_end)) {}

    [[nodiscard]] CStats computeStats(math::Matrix<size_t> const &confusion) {
      return {epoch, confusion, state};
    }

    [[nodiscard]] size_t getEpoch() const { return epoch; }
    void nextEpoch() { epoch++; }

  private:
    std::shared_ptr<CTrackerState> state;
    size_t epoch;
  };
}   // namespace control::classifier