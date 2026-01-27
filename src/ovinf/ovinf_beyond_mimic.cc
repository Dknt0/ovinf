#include "ovinf/ovinf_beyond_mimic.h"

namespace ovinf {

BeyondMimicPolicy::~BeyondMimicPolicy() {
  exiting_.store(true);
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

BeyondMimicPolicy::BeyondMimicPolicy(const YAML::Node &config)
    : BasePolicy(config) {
  // Read config
  size_t joint_counter = 0;
  for (auto const &name : config["policy_joint_names"]) {
    joint_names_[name.as<std::string>()] = joint_counter++;
  }

  single_obs_size_ = config["single_obs_size"].as<size_t>();
  current_obs_ = VectorT(single_obs_size_).setZero();
  action_size_ = config["action_size"].as<size_t>();
  if (action_size_ != joint_counter) {
    throw std::runtime_error("Action size mismatch");
  }
  action_scale_ = config["action_scale"].as<float>();
  obs_scale_ang_vel_ = config["obs_scales"]["ang_vel"].as<float>();
  obs_scale_command_ = config["obs_scales"]["command"].as<float>();
  obs_scale_dof_pos_ = config["obs_scales"]["dof_pos"].as<float>();
  obs_scale_dof_vel_ = config["obs_scales"]["dof_vel"].as<float>();
  clip_action_ = config["clip_action"].as<float>();
  joint_default_position_ = VectorT(joint_names_.size());
  stick_to_core_ = config["stick_to_core"].as<size_t>();
  log_name_ = config["log_name"].as<std::string>();

  for (auto const &pair : joint_names_) {
    joint_default_position_(pair.second, 0) =
        config["policy_default_position"][pair.first].as<float>();
  }

  // Create buffer
  last_action_ = VectorT(action_size_).setZero();
  latest_target_ = VectorT(action_size_).setZero();

  // Create logger
  log_flag_ = config["log_data"].as<bool>();
  if (log_flag_) {
    CreateLog(config);
  }

  // Create model
  auto model = ov::Core().read_model(model_path_);
  // compiled_model_ = ov::Core().compile_model(model_path_, device_);
  compiled_model_ = ov::Core().compile_model(model, device_);
  if (compiled_model_.input(0).get_element_type() != ov::element::f32) {
    throw std::runtime_error(
        "Model input type is not f32. Please convert the model to f32.");
  }

  // Get Trajectory Info
  dancing_started_.store(false);
  auto rt_info = model->get_rt_info();
  for (const auto &info : rt_info) {
    for (const auto &pair : info.second.as<ov::AnyMap>()) {
      if (pair.first == "traj_length") {
        traj_length_ = pair.second.as<size_t>();
      }
    }
  }
  std::cout << "Trajectory length from onnx rt_info: " << traj_length_
            << std::endl;
  std::cout << "Joint default position: " << joint_default_position_.transpose()
            << std::endl;

  infer_request_ = compiled_model_.create_infer_request();
  obs_info_ = compiled_model_.input(0);
  timestep_info_ = compiled_model_.input(1);

  ref_joint_pos_ = joint_default_position_;
  ref_joint_vel_ = VectorT::Zero(action_size_);
  ref_base_quat_ = QuaternionT::Identity();

  inference_done_.store(true);
  exiting_.store(false);
  worker_thread_ = std::thread(&BeyondMimicPolicy::WorkerThread, this);
}

bool BeyondMimicPolicy::WarmUp(RobotObservation<float> const &obs_pack) {
  VectorT obs(single_obs_size_);
  obs.setZero();
  timestep_input_ = 0.0;

  if (!inference_done_.load()) {
    return false;
  } else {
    current_obs_ = obs;

    ov::Tensor obs_tensor(obs_info_.get_element_type(), obs_info_.get_shape(),
                          current_obs_.data());
    ov::Tensor timestep_tensor(timestep_info_.get_element_type(),
                               timestep_info_.get_shape(), &timestep_input_);
    infer_start_time_ = std::chrono::high_resolution_clock::now();

    infer_request_.set_input_tensor(0, obs_tensor);
    infer_request_.set_input_tensor(1, timestep_tensor);
    inference_done_.store(false);

    return true;
  }
}

bool BeyondMimicPolicy::InferUnsync(RobotObservation<float> const &obs_pack) {
  VectorT obs(single_obs_size_);
  obs.setZero();

  if (obs_pack.command(0, 0) > 0.15f && !dancing_started_.load()) {
    dancing_started_.store(true);
    timestep_input_ = 0.0;
  } else if (timestep_input_ < traj_length_ - 2 && dancing_started_.load()) {
    timestep_input_ += 1.0;
  } else {
    timestep_input_ = 0.0;
    dancing_started_.store(false);
  }

  // Reference joint pos
  obs.segment(0, action_size_) = ref_joint_pos_;
  // Refenence joint vel
  obs.segment(action_size_, action_size_) = ref_joint_vel_;

  // Orientation difference
  // QuaternionT current_anchor_orientation =
  //     AngleAxisT(obs_pack.euler_angles(2), Vector3T::UnitZ()) *
  //     AngleAxisT(obs_pack.euler_angles(1), Vector3T::UnitY()) *
  //     AngleAxisT(obs_pack.euler_angles(0), Vector3T::UnitX());
  QuaternionT current_anchor_orientation =
      AngleAxisT(obs_pack.euler_angles(2), Vector3T::UnitZ()) *
      AngleAxisT(obs_pack.euler_angles(1), Vector3T::UnitY()) *
      AngleAxisT(obs_pack.euler_angles(0), Vector3T::UnitX());

  Matrix3T orientation_diff_matrix =
      (current_anchor_orientation.inverse() * ref_base_quat_)
          .toRotationMatrix();

  VectorT ori_input(6);
  ori_input(0, 0) = orientation_diff_matrix(0, 0);
  ori_input(1, 0) = orientation_diff_matrix(0, 1);
  ori_input(2, 0) = orientation_diff_matrix(1, 0);
  ori_input(3, 0) = orientation_diff_matrix(1, 1);
  ori_input(4, 0) = orientation_diff_matrix(2, 0);
  ori_input(5, 0) = orientation_diff_matrix(2, 1);
  obs.segment(2 * action_size_, 6) = ori_input;

  std::cout << ref_base_quat_.coeffs().transpose() << " ; "
            << current_anchor_orientation.coeffs().transpose() << std::endl;
  std::cout << ori_input.transpose() << std::endl;

  obs.segment(6 + 2 * action_size_, 3) = obs_pack.ang_vel;
  obs.segment(9 + 2 * action_size_, action_size_) =
      obs_pack.joint_pos - joint_default_position_;
  obs.segment(9 + 3 * action_size_, action_size_) = obs_pack.joint_vel;
  obs.segment(9 + 4 * action_size_, action_size_) = last_action_;

  if (!inference_done_.load()) {
    return false;
  } else {
    current_obs_ = obs;
    ov::Tensor obs_tensor(obs_info_.get_element_type(), obs_info_.get_shape(),
                          current_obs_.data());
    ov::Tensor timestep_tensor(timestep_info_.get_element_type(),
                               timestep_info_.get_shape(), &timestep_input_);
    infer_start_time_ = std::chrono::high_resolution_clock::now();

    infer_request_.set_input_tensor(0, obs_tensor);
    infer_request_.set_input_tensor(1, timestep_tensor);
    inference_done_.store(false);

    if (log_flag_) {
      WriteLog(obs_pack);
    }

    return true;
  }
}

std::optional<BeyondMimicPolicy::VectorT> BeyondMimicPolicy::GetResult(
    const size_t timeout) {
  if (inference_done_.load()) {
    return latest_target_;
  } else {
    std::this_thread::sleep_for(std::chrono::microseconds(timeout));
    if (inference_done_.load()) {
      return latest_target_;
    }
    return std::nullopt;
  }
}

void BeyondMimicPolicy::PrintInfo() {
  std::cout << "Load model: " << this->model_path_ << std::endl;
  std::cout << "Device: " << device_ << std::endl;
  std::cout << "Single obs size: " << single_obs_size_ << std::endl;
  std::cout << "Action size: " << action_size_ << std::endl;
  std::cout << "Action scale: " << action_scale_ << std::endl;
  std::cout << "  - Obs scale ang vel: " << obs_scale_ang_vel_ << std::endl;
  std::cout << "  - Obs scale command: " << obs_scale_command_ << std::endl;
  std::cout << "  - Obs scale dof pos: " << obs_scale_dof_pos_ << std::endl;
  std::cout << "  - Obs scale dof vel: " << obs_scale_dof_vel_ << std::endl;
  std::cout << "Clip action: " << clip_action_ << std::endl;
  std::cout << "Joint default position: " << joint_default_position_.transpose()
            << std::endl;
  std::cout << "Joint names: " << std::endl;
  for (const auto &pair : joint_names_) {
    std::cout << "  - " << pair.first << ": " << pair.second << std::endl;
  }
}

void BeyondMimicPolicy::WorkerThread() {
  if (realtime_) {
    if (!setProcessHighPriority(99)) {
      std::cerr << "Failed to set process high priority." << std::endl;
    }
    if (!StickThisThreadToCore(stick_to_core_)) {
      std::cerr << "Failed to stick thread to core." << std::endl;
    }
  }

  while (exiting_.load() == false) {
    if (!inference_done_.load()) {
      infer_request_.infer();
      auto action_tensor = infer_request_.get_output_tensor(0);
      auto ref_joint_pos_tensor = infer_request_.get_output_tensor(1);
      auto ref_joint_vel_tensor = infer_request_.get_output_tensor(2);
      auto body_quat_tensor = infer_request_.get_output_tensor(4);

      infer_end_time_ = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_seconds =
          infer_end_time_ - infer_start_time_;
      // std::cout << "Inference time: " << elapsed_seconds.count() * 1000
      //           << "ms" << std::endl;
      inference_time_ = elapsed_seconds.count() * 1000;

      ref_joint_pos_ =
          Eigen::Map<VectorT>(ref_joint_pos_tensor.data<float>(), action_size_);
      ref_joint_vel_ =
          Eigen::Map<VectorT>(ref_joint_vel_tensor.data<float>(), action_size_);
      ref_base_quat_ = QuaternionT(
          body_quat_tensor.data<float>()[0], body_quat_tensor.data<float>()[1],
          body_quat_tensor.data<float>()[2], body_quat_tensor.data<float>()[3]);

      VectorT action_eigen =
          Eigen::Map<VectorT>(action_tensor.data<float>(), action_size_)
              .cwiseMin(clip_action_)
              .cwiseMax(-clip_action_);

      last_action_ = action_eigen;
      latest_target_ = action_eigen * action_scale_ + joint_default_position_;
    }
    inference_done_.store(true);
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

void BeyondMimicPolicy::CreateLog(YAML::Node const &config) {
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm *now_tm = std::localtime(&now_time);
  std::stringstream ss;
  ss << std::put_time(now_tm, "%Y-%m-%d-%H-%M-%S");
  std::string current_time = ss.str();

  std::string log_dir = config["log_dir"].as<std::string>();
  std::filesystem::path config_file_path(log_dir);
  if (config_file_path.is_relative()) {
    config_file_path = canonical(config_file_path);
  }

  if (!exists(config_file_path)) {
    create_directories(config_file_path);
  }

  std::string logger_file = config_file_path.string() + "/" + current_time +
                            "_humanoid_" + log_name_ + ".csv";

  // Get headers
  std::vector<std::string> headers;

  headers.push_back("command_vel_x");
  headers.push_back("command_vel_y");
  headers.push_back("command_vel_w");
  for (size_t i = 0; i < action_size_; ++i) {
    headers.push_back("joint_pos_" + std::to_string(i));
  }
  for (size_t i = 0; i < action_size_; ++i) {
    headers.push_back("joint_vel_" + std::to_string(i));
  }
  for (size_t i = 0; i < action_size_; ++i) {
    headers.push_back("last_action_" + std::to_string(i));
  }
  headers.push_back("ang_vel_x");
  headers.push_back("ang_vel_y");
  headers.push_back("ang_vel_z");
  headers.push_back("prog_gravity_x");
  headers.push_back("prog_gravity_y");
  headers.push_back("prog_gravity_z");
  headers.push_back("inference_time_ms");

  csv_logger_ = std::make_shared<CsvLogger>(logger_file, headers);
}

void BeyondMimicPolicy::WriteLog(RobotObservation<float> const &obs_pack) {
  std::vector<CsvLogger::Number> datas;

  for (size_t i = 0; i < 3; ++i) {
    datas.push_back(obs_pack.command(i));
  }
  for (size_t i = 0; i < action_size_; ++i) {
    datas.push_back(obs_pack.joint_pos(i));
  }
  for (size_t i = 0; i < action_size_; ++i) {
    datas.push_back(obs_pack.joint_vel(i));
  }
  for (size_t i = 0; i < action_size_; ++i) {
    datas.push_back(last_action_(i));
  }
  for (size_t i = 0; i < 3; ++i) {
    datas.push_back(obs_pack.ang_vel(i));
  }
  for (size_t i = 0; i < 3; ++i) {
    datas.push_back(obs_pack.proj_gravity(i));
  }
  datas.push_back(inference_time_);

  csv_logger_->Write(datas);
}

}  // namespace ovinf
