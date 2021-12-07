
#include "classifierTracker.hpp"

namespace control::classifier {

  CStats::CStats(size_t epoch, const math::Matrix<size_t> &confusion,
                 std::shared_ptr<CTrackerState> state)
      : epoch(epoch), avg_f1score(0), avg_precision(0), avg_recall(0), state(std::move(state)) {
    stats = math::Matrix<double>(confusion.getRows(), 3);

    stats.fill(0);
    for (size_t i = 0; i < confusion.getRows(); i++) {
      size_t sum = 0;

      // Compute the precision
      for (size_t j = 0; j < confusion.getCols(); j++) { sum += confusion(i, j); }
      stats(i, 0) = static_cast<double>(confusion(i, i)) / static_cast<double>(sum);

      // Compute the recall
      sum = 0;
      for (size_t j = 0; j < confusion.getRows(); j++) { sum += confusion(j, i); }
      stats(i, 1) = static_cast<double>(confusion(i, i)) / static_cast<double>(sum);

      // The f1 score is the harmonic mean of both the recall and the precision
      stats(i, 2) = 2 * (stats(i, 0) * stats(i, 1)) / (stats(i, 0) + stats(i, 1));

      for (size_t j = 0; j < 3; j++)
        if (std::isnan(stats(i, j))) stats(i, j) = 0.0;
      avg_precision += stats(i, 0);
      avg_recall += stats(i, 1);
      avg_f1score += stats(i, 2);
    }

    const auto nclass = static_cast<double>(confusion.getRows());

    // Compute the avg of each stat
    avg_precision /= nclass;
    avg_recall /= nclass;
    avg_f1score /= nclass;
  }

  std::ostream &operator<<(std::ostream &os, CStats const &stat) {
    os << "[epoch: " << stat.epoch << "] avg_precision: " << stat.avg_precision
       << ", avg_recall: " << stat.avg_recall << ", avg_f1score: " << stat.avg_f1score << std::endl;
    return os;
  }

  void CStats::dumpToFiles() {
    size_t const nclass = stats.getRows();

    CTrackerState &s = *state;
    for (size_t i = 0; i < nclass; i++) {
      s.getPrecOutput(i) << epoch << " " << stats(i, 0) << std::endl;
      s.getRecallOutput(i) << epoch << " " << stats(i, 1) << std::endl;
      s.getF1Output(i) << epoch << " " << stats(i, 2) << std::endl;
    }

    s.getAvgPrecOutput() << epoch << " " << avg_precision << std::endl;
    s.getAvgRecallOutput() << epoch << " " << avg_recall << std::endl;
    s.getAvgF1Output() << epoch << " " << avg_f1score << std::endl;
  }

  void CStats::classification_report(std::ostream &os,
                                     std::vector<ClassLabel> const &labels) const {
    size_t nclass = stats.getRows();

    for (size_t i = 0; i < nclass; i++) {
      os << "[" << labels[i] << "] precision: " << stats(i, 0) << ", recall: " << stats(i, 1)
         << ", f1-score: " << stats(i, 2) << std::endl;
    }
    os << "average accuracy: " << avg_precision << std::endl;
    os << "average recall: " << avg_recall << std::endl;
    os << "average f1-score: " << avg_f1score << std::endl;
  }

}   // namespace control::classifier