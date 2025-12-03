#ifndef OVINF_HPP
#define OVINF_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <filesystem>
#include <openvino/openvino.hpp>

namespace ovinf {

template <typename T = float>
struct RobotObservation {
  Eigen::Matrix<T, Eigen::Dynamic, 1> clock = {};
  Eigen::Matrix<T, Eigen::Dynamic, 1> command = {};
  Eigen::Matrix<T, Eigen::Dynamic, 1> ang_vel = {};
  Eigen::Matrix<T, Eigen::Dynamic, 1> proj_gravity = {};
  Eigen::Matrix<T, Eigen::Dynamic, 1> joint_pos = {};
  Eigen::Matrix<T, Eigen::Dynamic, 1> joint_vel = {};
  Eigen::Matrix<T, Eigen::Dynamic, 1> last_action = {};
  Eigen::Matrix<T, Eigen::Dynamic, 1> position = {};
  Eigen::Matrix<T, Eigen::Dynamic, 1> velocity = {};
  Eigen::Matrix<T, Eigen::Dynamic, 1> euler_angles = {};
  Eigen::Matrix<T, Eigen::Dynamic, 1> scan = {};
  Eigen::Matrix<T, Eigen::Dynamic, 1> custom = {};
};

template <typename T = float>
class BasePolicy {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

 public:
  using BasePolicyPtr = std::shared_ptr<BasePolicy<T>>;

  BasePolicy() = delete;
  BasePolicy(const YAML::Node &config) {
    device_ = config["device"].as<std::string>();
    model_path_ = config["model_path"].as<std::string>();
    realtime_ = config["realtime"].as<bool>();
  }
  virtual ~BasePolicy() = default;

  /**
   * @brief Policy warmup
   *
   * @param[in] obs_pack Robot observation
   * @param[in] num_itrations Warmup iterations
   * @return Is warmup done successfully.
   */
  virtual bool WarmUp(const RobotObservation<T> &obs) = 0;

  /**
   * @brief Set observation, run inference.
   *
   * @param[in] obs_pack Robot observation
   * @return Is inference started immidiately.
   */
  [[nodiscard("Return value of InferUnsync should be checked.")]] virtual bool
  InferUnsync(const RobotObservation<T> &obs) = 0;

  /**
   * @brief Get resulting target_joint_pos
   *
   * @param[in] timeout Timeout in microseconds
   */
  virtual std::optional<Eigen::Matrix<T, Eigen::Dynamic, 1>> GetResult(
      const size_t timeout = 100) = 0;

  virtual void PrintInfo() = 0;

 protected:
  std::string device_;
  std::string model_path_;
  bool realtime_;
};

}  // namespace ovinf

#endif  // !OVINF_HPP
