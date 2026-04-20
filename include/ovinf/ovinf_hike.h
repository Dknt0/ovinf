/**
 * @file ovinf_hike.h
 * @brief Clock-free
 * @author Dknt
 * @date 2025-7
 */

#ifndef OVINF_HIKE_H
#define OVINF_HIKE_H

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <openvino/openvino.hpp>
#include <optional>
#include <string>
#include <thread>

#include "atomicops.h"
#include "ovinf.hpp"
#include "readerwriterqueue.h"
#include "utils/csv_logger.hpp"
#include "utils/history_buffer.hpp"
#include "utils/realtime_setting.h"
#include "utils/skipped_ring_buffer.hpp"

namespace ovinf {

/**
 * @brief clock-free inference.
 */
class HikePolicy : public BasePolicy<float> {
  using MatrixT = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorT = Eigen::Matrix<float, Eigen::Dynamic, 1>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

 public:
  HikePolicy() = delete;
  ~HikePolicy();

  HikePolicy(const YAML::Node &config);

  /**
   * @brief Policy warmup
   *
   * @param[in] obs_pack Robot observation
   * @param[in] num_itrations Warmup iterations
   * @return Is warmup done successfully.
   */
  virtual bool WarmUp(RobotObservation<float> const &obs_pack) final;

  /**
   * @brief Set observation, run inference.
   *
   * @param[in] obs_pack Robot observation
   * @return Is inference started immidiately.
   */
  virtual bool InferUnsync(RobotObservation<float> const &obs_pack) final;

  /**
   * @brief Get resulting target_joint_pos
   *
   * @param[in] timeout Timeout in microseconds
   */
  virtual std::optional<VectorT> GetResult(const size_t timeout = 100) final;

  virtual void PrintInfo() final;

 private:
  void WorkerThread();
  void CreateLog(YAML::Node const &config);
  void WriteLog(RobotObservation<float> const &obs_pack);

 private:
  // Threading
  std::atomic<bool> inference_done_{false};
  std::atomic<bool> exiting_{false};
  std::thread worker_thread_;

  // OpenVINO inference
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;
  ov::Output<const ov::Node> input_info_;

  // Infer data
  std::map<std::string, size_t> joint_names_;
  VectorT joint_default_position_;

  size_t single_obs_size_;
  size_t obs_buffer_size_;
  size_t action_size_;

  float action_scale_;
  float obs_scale_ang_vel_;
  float obs_scale_lin_vel_;
  float obs_scale_command_;
  float obs_scale_dof_pos_;
  float obs_scale_dof_vel_;
  float obs_scale_proj_gravity_;
  float clip_action_;

  size_t num_images_ = 8;
  size_t image_skip_ = 5;
  size_t image_width_ = 32;
  size_t image_height_ = 18;

  // Buffer
  moodycamel::ReaderWriterQueue<VectorT> input_queue_;
  std::shared_ptr<HistoryBuffer<float>> obs_buffer_;
  std::shared_ptr<SkippedRingBuffer> depth_skipped_buffer_;

  VectorT policy_input_vec_;

  VectorT last_action_;
  VectorT latest_target_;

  // Clock
  std::chrono::high_resolution_clock::time_point infer_start_time_;
  std::chrono::high_resolution_clock::time_point infer_end_time_;

  // Logger
  bool log_flag_ = false;
  CsvLogger::Ptr csv_logger_;
  std::string log_name_;
  float inference_time_ = 0;

  // Realtime
  size_t stick_to_core_ = 0;
};
}  // namespace ovinf

#endif  // !OVINF_HIKE_H
