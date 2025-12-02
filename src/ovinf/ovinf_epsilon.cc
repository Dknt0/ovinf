#include "ovinf/ovinf_epsilon.h"

namespace ovinf {

EpsilonPolicy::~EpsilonPolicy() {
  exiting_.store(true);
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

EpsilonPolicy::EpsilonPolicy(const YAML::Node &config) : BasePolicy(config) {
  // Read config
  size_t joint_counter = 0;
  for (auto const &name : config["policy_joint_names"]) {
    joint_names_[name.as<std::string>()] = joint_counter++;
  }

  single_obs_size_ = config["single_obs_size"].as<size_t>();
  obs_buffer_size_ = config["obs_buffer_size"].as<size_t>();
  action_size_ = config["action_size"].as<size_t>();
  if (action_size_ != joint_counter) {
    throw std::runtime_error("Action size mismatch");
  }
  action_scale_ = config["action_scale"].as<float>();
  obs_scale_ang_vel_ = config["obs_scales"]["ang_vel"].as<float>();
  obs_scale_lin_vel_ = config["obs_scales"]["lin_vel"].as<float>();
  obs_scale_command_ = config["obs_scales"]["command"].as<float>();
  obs_scale_dof_pos_ = config["obs_scales"]["dof_pos"].as<float>();
  obs_scale_dof_vel_ = config["obs_scales"]["dof_vel"].as<float>();
  obs_scale_proj_gravity_ = config["obs_scales"]["proj_gravity"].as<float>();
  clip_action_ = config["clip_action"].as<float>();
  joint_default_position_ = VectorT(joint_names_.size());
  stick_to_core_ = config["stick_to_core"].as<size_t>();
  log_name_ = config["log_name"].as<std::string>();

  for (auto const &pair : joint_names_) {
    joint_default_position_(pair.second, 0) =
        config["policy_default_position"][pair.first].as<float>();
  }

  // Create buffer
  obs_buffer_ = std::make_shared<HistoryBuffer<float>>(single_obs_size_,
                                                       obs_buffer_size_);
  input_queue_ = moodycamel::ReaderWriterQueue<VectorT>(obs_buffer_size_ * 2);
  last_action_ = VectorT(action_size_).setZero();
  latest_target_ = VectorT(action_size_).setZero();

  // Create logger
  log_flag_ = config["log_data"].as<bool>();
  if (log_flag_) {
    CreateLog(config);
  }

  // Create model
  compiled_model_ = ov::Core().compile_model(model_path_, device_);
  if (compiled_model_.input().get_element_type() != ov::element::f32) {
    throw std::runtime_error(
        "Model input type is not f32. Please convert the model to f32.");
  }

  infer_request_ = compiled_model_.create_infer_request();
  input_info_ = compiled_model_.input();

  inference_done_.store(true);
  exiting_.store(false);
  worker_thread_ = std::thread(&EpsilonPolicy::WorkerThread, this);
}

bool EpsilonPolicy::WarmUp(RobotObservation<float> const &obs_pack) {
  double gait_time_value = 0.0;

  VectorT obs(single_obs_size_);
  obs.setZero();

  if (!inference_done_.load()) {
    input_queue_.enqueue(obs);
    return false;
  } else {
    while (input_queue_.peek() != nullptr) {
      VectorT old_obs;
      input_queue_.try_dequeue(old_obs);
      obs_buffer_->AddObservation(old_obs);
    }
    obs_buffer_->AddObservation(obs);

    ov::Tensor input_tensor(input_info_.get_element_type(),
                            input_info_.get_shape(),
                            obs_buffer_->GetObsHistory().data());
    infer_start_time_ = std::chrono::high_resolution_clock::now();

    infer_request_.set_input_tensor(input_tensor);
    inference_done_.store(false);

    return true;
  }
}

bool EpsilonPolicy::InferUnsync(RobotObservation<float> const &obs_pack) {
  VectorT obs(single_obs_size_);
  obs.setZero();
  VectorT command_scaled(3);
  command_scaled.segment(0, 2) =
      obs_pack.command.segment(0, 2) * obs_scale_lin_vel_;
  command_scaled(2) = obs_pack.command(2) * obs_scale_ang_vel_;
  obs.segment(0, 3) = command_scaled * obs_scale_command_;
  obs.segment(3, action_size_) =
      (obs_pack.joint_pos - joint_default_position_) * obs_scale_dof_pos_;
  obs.segment(3 + action_size_, action_size_) =
      obs_pack.joint_vel * obs_scale_dof_vel_;
  obs.segment(3 + 2 * action_size_, action_size_) = last_action_;
  obs.segment(3 + 3 * action_size_, 3) = obs_pack.ang_vel * obs_scale_ang_vel_;
  obs.segment(6 + 3 * action_size_, 3) =
      obs_pack.proj_gravity * obs_scale_proj_gravity_;

  if (!inference_done_.load()) {
    input_queue_.enqueue(obs);
    return false;
  } else {
    while (input_queue_.peek() != nullptr) {
      VectorT old_obs;
      input_queue_.try_dequeue(old_obs);
      obs_buffer_->AddObservation(old_obs);
    }
    obs_buffer_->AddObservation(obs);

    ov::Tensor input_tensor(input_info_.get_element_type(),
                            input_info_.get_shape(),
                            obs_buffer_->GetObsHistory().data());
    infer_start_time_ = std::chrono::high_resolution_clock::now();

    infer_request_.set_input_tensor(input_tensor);
    inference_done_.store(false);

    if (log_flag_) {
      WriteLog(obs_pack);
    }

    return true;
  }
}

std::optional<EpsilonPolicy::VectorT> EpsilonPolicy::GetResult(
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

void EpsilonPolicy::PrintInfo() {
  std::cout << "Load model: " << this->model_path_ << std::endl;
  std::cout << "Device: " << device_ << std::endl;
  std::cout << "Single obs size: " << single_obs_size_ << std::endl;
  std::cout << "Obs buffer size: " << obs_buffer_size_ << std::endl;
  std::cout << "Action size: " << action_size_ << std::endl;
  std::cout << "Action scale: " << action_scale_ << std::endl;
  std::cout << "  - Obs scale ang vel: " << obs_scale_ang_vel_ << std::endl;
  std::cout << "  - Obs scale lin vel: " << obs_scale_lin_vel_ << std::endl;
  std::cout << "  - Obs scale command: " << obs_scale_command_ << std::endl;
  std::cout << "  - Obs scale dof pos: " << obs_scale_dof_pos_ << std::endl;
  std::cout << "  - Obs scale dof vel: " << obs_scale_dof_vel_ << std::endl;
  std::cout << "  - Obs scale proj gravity: " << obs_scale_proj_gravity_
            << std::endl;
  std::cout << "Clip action: " << clip_action_ << std::endl;
  std::cout << "Joint default position: " << joint_default_position_.transpose()
            << std::endl;
  std::cout << "Joint names: " << std::endl;
  for (const auto &pair : joint_names_) {
    std::cout << "  - " << pair.first << ": " << pair.second << std::endl;
  }
}

void EpsilonPolicy::WorkerThread() {
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
      auto action_tensor = infer_request_.get_output_tensor();
      infer_end_time_ = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_seconds =
          infer_end_time_ - infer_start_time_;
      // std::cout << "Inference time: " << elapsed_seconds.count() * 1000
      //           << "ms" << std::endl;
      inference_time_ = elapsed_seconds.count() * 1000;

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

void EpsilonPolicy::CreateLog(YAML::Node const &config) {
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

void EpsilonPolicy::WriteLog(RobotObservation<float> const &obs_pack) {
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
