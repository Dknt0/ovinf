#ifndef ROBOT_BASE_HPP
#define ROBOT_BASE_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <map>
#include <string>

namespace ovinf {

template <typename T = float>
class RobotBase {
  using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  friend class ObserverBase;
  friend class ExecutorBase;

 public:
  using RobotPtr = std::shared_ptr<RobotBase<T>>;

  class ObserverBase {
   public:
    using ObserverPtr = std::shared_ptr<ObserverBase>;

    ObserverBase() = delete;
    ObserverBase(RobotBase<T>* robot, const YAML::Node& config) {
      robot_ = robot;
      motor_size_ = robot->motor_size_;
      joint_size_ = robot->joint_size_;

      motor_actual_position_ = VectorT::Zero(motor_size_);
      motor_actual_velocity_ = VectorT::Zero(motor_size_);
      motor_actual_current_ = VectorT::Zero(motor_size_);
      joint_actual_position_ = VectorT::Zero(joint_size_);
      joint_actual_velocity_ = VectorT::Zero(joint_size_);
      angular_velocity_ = VectorT::Zero(3);
      acceleration_ = VectorT::Zero(3);
      euler_rpy_ = VectorT::Zero(3);
      proj_gravity_ = VectorT::Zero(3);
    }

    virtual bool Update() = 0;

    const VectorT& MotorActualPosition() { return motor_actual_position_; }
    const VectorT& MotorActualVelocity() { return motor_actual_velocity_; }
    const VectorT& MotorActualCurrent() { return motor_actual_current_; }
    const VectorT& JointActualPosition() { return joint_actual_position_; }
    const VectorT& JointActualVelocity() { return joint_actual_velocity_; }

    const VectorT& AngularVelocity() { return angular_velocity_; }
    const VectorT& Acceleration() { return acceleration_; }
    const VectorT& EulerRpy() { return euler_rpy_; }
    const VectorT& ProjGravity() { return proj_gravity_; }

    const VectorT& Scan() { return scan_; }

   protected:
    size_t motor_size_ = 0;
    size_t joint_size_ = 0;

    RobotBase<T>* robot_;
    VectorT motor_actual_position_;
    VectorT motor_actual_velocity_;
    VectorT motor_actual_current_;
    VectorT joint_actual_position_;
    VectorT joint_actual_velocity_;

    VectorT angular_velocity_;
    VectorT acceleration_;
    VectorT euler_rpy_;
    VectorT proj_gravity_;

    VectorT scan_;
  };

  class ExecutorBase {
   public:
    using ExecutorPtr = std::shared_ptr<ExecutorBase>;

    ExecutorBase() = delete;
    ExecutorBase(RobotBase<T>* robot, const YAML::Node& config) {
      robot_ = robot;
      motor_size_ = robot->motor_size_;
      joint_size_ = robot->joint_size_;

      motor_target_position_ = VectorT::Zero(motor_size_);
      motor_target_torque_ = VectorT::Zero(motor_size_);
      motor_target_current_ = VectorT::Zero(motor_size_);
      motor_target_p_gain_ = VectorT::Zero(motor_size_);
      motor_target_d_gain_ = VectorT::Zero(motor_size_);
      joint_target_position_ = VectorT::Zero(motor_size_);
      joint_target_torque_ = VectorT::Zero(motor_size_);
      joint_target_p_gain_ = VectorT::Zero(motor_size_);
      joint_target_d_gain_ = VectorT::Zero(motor_size_);

      torque_limit_ = VectorT::Zero(motor_size_);
      motor_upper_limit_ = VectorT::Zero(motor_size_);
      motor_lower_limit_ = VectorT::Zero(motor_size_);

      // Read from config
      for (auto const& pair : robot_->motor_names_) {
        torque_limit_(pair.second, 0) =
            config["torque_limit"][pair.first].template as<float>();
        motor_upper_limit_(pair.second, 0) =
            config["motor_upper_limit"][pair.first].template as<float>();
        motor_lower_limit_(pair.second, 0) =
            config["motor_lower_limit"][pair.first].template as<float>();
      }
    }

    virtual bool ExecuteJointTorque() = 0;
    virtual bool ExecuteJointPosition() { return false; }
    virtual bool ExecuteMotorTorque() = 0;
    virtual bool ExecuteMotorCurrent() = 0;
    virtual bool ExecuteMotorPosition() { return false; }

    VectorT& MotorTargetPosition() { return motor_target_position_; }
    VectorT& MotorTargetTorque() { return motor_target_torque_; }
    VectorT& MotorTargetCurrent() { return motor_target_current_; }
    VectorT& MotorTargetPGain() { return motor_target_p_gain_; }
    VectorT& MotorTargetDGain() { return motor_target_d_gain_; }
    VectorT& JointTargetPosition() { return joint_target_position_; }
    VectorT& JointTargetTorque() { return joint_target_torque_; }
    VectorT& JointTargetPGain() { return joint_target_p_gain_; }
    VectorT& JointTargetDGain() { return joint_target_d_gain_; }

    const VectorT& TorqueLimit() { return torque_limit_; }
    const VectorT& MotorUpperLimit() { return motor_upper_limit_; }
    const VectorT& MotorLowerLimit() { return motor_lower_limit_; }

   protected:
    size_t motor_size_ = 0;
    size_t joint_size_ = 0;

    VectorT torque_limit_;
    VectorT motor_upper_limit_;
    VectorT motor_lower_limit_;

    RobotBase<T>* robot_;
    VectorT motor_target_position_;
    VectorT motor_target_torque_;
    VectorT motor_target_current_;
    VectorT motor_target_p_gain_;
    VectorT motor_target_d_gain_;
    VectorT joint_target_position_;
    VectorT joint_target_torque_;
    VectorT joint_target_p_gain_;
    VectorT joint_target_d_gain_;
  };

 public:
  RobotBase() = delete;
  RobotBase(const YAML::Node& config) {
    robot_name_ = config["robot_name"].as<std::string>();
    size_t joint_counter = 0;
    for (auto const& name : config["joint_names"]) {
      joint_names_[name.as<std::string>()] = joint_counter++;
    }
    size_t motor_counter = 0;
    for (auto const& name : config["motor_names"]) {
      motor_names_[name.as<std::string>()] = motor_counter++;
    }
    joint_size_ = config["joint_size"].as<size_t>();
    motor_size_ = config["motor_size"].as<size_t>();

    if (joint_size_ != joint_names_.size()) {
      throw std::runtime_error(
          std::string("Joint size not match") +
          " joint size: " + std::to_string(joint_size_) +
          ", joint names size: " + std::to_string(joint_names_.size()));
    } else if (motor_size_ != motor_names_.size()) {
      throw std::runtime_error(
          std::string("Motor size not match") +
          " motor size: " + std::to_string(motor_size_) +
          ", motor names size: " + std::to_string(motor_names_.size()));
    }

    motor_direction_ = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(motor_size_);

    for (auto const& pair : motor_names_) {
      motor_direction_(pair.second, 0) =
          config["motor_direction"][pair.first].as<int>();
    }
  }

  virtual void PrintInfo() = 0;

  ObserverBase::ObserverPtr Observer() { return observer_; }
  ExecutorBase::ExecutorPtr Executor() { return executor_; }

 public:
  std::string robot_name_;
  std::map<std::string, size_t> joint_names_;  // joint name to index
  std::map<std::string, size_t> motor_names_;  // motor name to index
  Eigen::Matrix<int, Eigen::Dynamic, 1> motor_direction_;
  size_t joint_size_ = 0;
  size_t motor_size_ = 0;

 protected:
  typename ObserverBase::ObserverPtr observer_;
  typename ExecutorBase::ExecutorPtr executor_;
};

}  // namespace ovinf

#endif  // !ROBOT_BASE_HPP
