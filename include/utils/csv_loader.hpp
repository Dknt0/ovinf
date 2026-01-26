#ifndef CSV_LOADER_HPP
#define CSV_LOADER_HPP

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace ovinf {

template <typename T = float>
class TrajectoryLoader {
 public:
  using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  TrajectoryLoader() = delete;

  TrajectoryLoader(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open file: " + file_path);
    }

    std::string line;
    std::vector<std::vector<T>> data;
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;
      std::vector<T> row;
      while (std::getline(ss, value, ',')) {
        row.push_back(std::stof(value));
      }
      data.push_back(row);
    }
    file.close();

    if (data.empty()) {
      throw std::runtime_error("No data found in file: " + file_path);
    }

    size_t rows = data.size();
    size_t cols = data[0].size();
    trajectory_.resize(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      if (data[i].size() != cols) {
        throw std::runtime_error("Inconsistent number of columns in file: " +
                                 file_path);
      }
      for (size_t j = 0; j < cols; ++j) {
        trajectory_(i, j) = data[i][j];
      }
    }
  }
  MatrixT const &GetTrajectory() const { return trajectory_; }

 private:
  MatrixT trajectory_;
};

}  // namespace ovinf

#endif
