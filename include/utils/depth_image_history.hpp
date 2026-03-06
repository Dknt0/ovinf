#ifndef DEPTH_IMAGE_HISTORY_HPP
#define DEPTH_IMAGE_HISTORY_HPP

#include <Eigen/Core>
// #include <iostream>
#include <stdexcept>
#include <vector>

namespace bitbot {

/**
 * @brief Circular buffer for storing depth image history (Eigen only).
 *        Supports retrieval with history skipping for encoder input.
 *        Does NOT support multi-threading!
 */
class DepthImageHistory {
 public:
  using DepthImageFlatten = Eigen::Vector<float, Eigen::Dynamic>;

  /**
   * @brief Constructor
   * @param buffer_size Maximum number of images to store in buffer
   */
  DepthImageHistory(size_t buffer_size)
      : buffer_size_(buffer_size), current_index_(-1) {
    images_.resize(buffer_size);
  }

  /**
   * @brief Add new depth image to history
   * @param img Eigen matrix (depth image)
   */
  void AddImage(const DepthImageFlatten &img) {
    if (current_index_ == -1) [[unlikely]] {
      // std::cout << "Resetting in AddImage." << std::endl;
      Reset(img);
    } else {
      images_[current_index_] = img;
      current_index_ = (current_index_ + 1) % buffer_size_;
    }
  }

  /**
   * @brief Retrieve depth image history for encoder, with skipping.
   *        E.g. [t-k*skip, ..., t-skip, t] for num_history=k+1, skip=s
   * @param num_history Number of images in history (e.g. 9)
   * @param skip History skipping (e.g. 8 -> t-8*skip,...,t)
   * @return std::vector<DepthImage> Oldest first, newest last
   */
  DepthImageFlatten GetHistory(size_t num_history, size_t skip) const {
    size_t available = buffer_size_;
    if (num_history < 1) throw std::invalid_argument("num_history must be > 0");
    if ((num_history - 1) * skip >= available)
      throw std::out_of_range(
          "Not enough history in buffer for requested skipping.");
    DepthImageFlatten result(num_history * images_[0].size());

    // Start from newest, sample with skip backwards. The newest image is at the
    // end of the result
    for (size_t i = 0; i < num_history; ++i) {
      size_t idx =
          (current_index_ + buffer_size_ - i * skip - 1) % buffer_size_;
      result.segment((num_history - i - 1) * images_[0].size(),
                     images_[0].size()) = images_[idx];
    }
    return result;
  }

  /**
   * @brief Reset buffer
   */
  // Fill buffer with a provided image after reset
  void Reset(const DepthImageFlatten &img) {
    for (auto &buf_img : images_) {
      buf_img = img;
    }
    current_index_ = 0;
  }

  /**
   * @brief Retrieve single depth image at given history index
   * @param index Index (0 = most recent, buffer_size_-1 = oldest)
   * @return Depth image at index
   */
  const DepthImageFlatten &GetImage(size_t index) const {
    if (index >= buffer_size_) throw std::out_of_range("Index out of range.");
    size_t idx = (current_index_ + buffer_size_ - index - 1) % buffer_size_;
    return images_[idx];
  }

  /**
   * @brief Current number of stored images
   */
  size_t Size() const { return buffer_size_; }

 private:
  std::vector<DepthImageFlatten> images_;  // Circular buffer
  size_t buffer_size_;
  size_t current_index_;
};

}  // namespace bitbot

#endif  // DEPTH_IMAGE_HISTORY_HPP
