#ifndef SKIPPED_RING_BUFFER_HPP
#define SKIPPED_RING_BUFFER_HPP

#include <Eigen/Core>
// #include <iostream>
#include <stdexcept>
#include <vector>

namespace ovinf {

/**
 * @brief Circular buffer for storing history of any Eigen::Vector data type
 * (Eigen only). Supports retrieval with history skipping for encoder input.
 *        Does NOT support multi-threading!
 */
class SkippedRingBuffer {
 public:
  using ValueType = Eigen::Vector<float, Eigen::Dynamic>;

  /**
   * @brief Constructor
   * @param buffer_size Maximum number of values to store in buffer
   */
  SkippedRingBuffer(size_t buffer_size)
      : buffer_size_(buffer_size), current_index_(-1) {
    buffer_.resize(buffer_size);
  }

  /**
   * @brief Add new value to history
   * @param value Eigen vector
   */
  void Add(const ValueType &value) {
    if (current_index_ == -1) [[unlikely]] {
      // std::cout << "Resetting in Add." << std::endl;
      Reset(value);
    } else {
      buffer_[current_index_] = value;
      current_index_ = (current_index_ + 1) % buffer_size_;
    }
  }

  /**
   * @brief Retrieve history for encoder, with skipping.
   *        E.g. [t-k*skip, ..., t-skip, t] for num_history=k+1, skip=s
   * @param num_history Number of entries in history (e.g. 9)
   * @param skip History skipping (e.g. 8 -> t-8*skip,...,t)
   * @return ValueType Oldest first, newest last
   */
  ValueType GetHistory(size_t num_history, size_t skip) const {
    size_t available = buffer_size_;
    if (num_history < 1) throw std::invalid_argument("num_history must be > 0");
    if ((num_history - 1) * skip >= available)
      throw std::out_of_range(
          "Not enough history in buffer for requested skipping.");
    ValueType result(num_history * buffer_[0].size());

    // Start from newest, sample with skip backwards. The newest is last in
    // result
    for (size_t i = 0; i < num_history; ++i) {
      size_t idx =
          (current_index_ + buffer_size_ - i * skip - 1) % buffer_size_;
      result.segment((num_history - i - 1) * buffer_[0].size(),
                     buffer_[0].size()) = buffer_[idx];
    }
    return result;
  }

  /**
   * @brief Reset buffer
   */
  // Fill buffer with a provided value after reset
  void Reset(const ValueType &value) {
    for (auto &elem : buffer_) {
      elem = value;
    }
    current_index_ = 0;
  }

  /**
   * @brief Retrieve single value at given history index
   * @param index Index (0 = most recent, buffer_size_-1 = oldest)
   * @return Value at index
   */
  const ValueType &Get(size_t index) const {
    if (index >= buffer_size_) throw std::out_of_range("Index out of range.");
    size_t idx = (current_index_ + buffer_size_ - index - 1) % buffer_size_;
    return buffer_[idx];
  }

  /**
   * @brief Current number of stored values
   */
  size_t Size() const { return buffer_size_; }

 private:
  std::vector<ValueType> buffer_;  // Circular buffer
  size_t buffer_size_;
  size_t current_index_;
};

}  // namespace ovinf

#endif  // SKIPPED_RING_BUFFER_HPP
