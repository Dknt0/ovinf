#ifndef FILTER_LOW_PASS_HPP
#define FILTER_LOW_PASS_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>

#include "filter/filter_base.hpp"

namespace ovinf {

template <typename T = Eigen::Matrix<float, Eigen::Dynamic, 1>>
class LowPassFilter : public FilterBase<T> {
 public:
  LowPassFilter() = delete;
  LowPassFilter(YAML::Node const &config) : FilterBase<T>(config) {
    lower_bound_ = this->ReadYamlParam(config["lower_bound"]);
    upper_bound_ = this->ReadYamlParam(config["upper_bound"]);
    alpha_ = this->ReadYamlParam(config["alpha"]);
    init_flag_ = false;

    if constexpr (is_eigen_vector_v<T>) {
      this->dimension_ = lower_bound_.rows();
      this->last_input_ = T(this->dimension_).setZero();
      this->last_output_ = T(this->dimension_).setZero();

      if (lower_bound_.size() != this->dimension_ ||
          upper_bound_.size() != this->dimension_ ||
          alpha_.size() != this->dimension_) {
        throw std::runtime_error(
            "PassThroughFilter: lower_bound or upper_bound size doesn't match "
            "the vector dimension. "
            "Expected dimension: " +
            std::to_string(this->dimension_));
      }
      one_minus_alpha_ = T(this->dimension_).setOnes() - alpha_;
    } else {
      this->last_input_ = 0;
      this->last_output_ = 0;
      one_minus_alpha_ = 1.0 - alpha_;
    }
  }

  virtual T Filter(T const &input) final {
    if constexpr (is_eigen_vector_v<T>) {
      if (!init_flag_) [[unlikely]] {
        init_flag_ = true;
        last_output_ = this->NanHandle(input)
                           .cwiseMin(upper_bound_)
                           .cwiseMax(lower_bound_);
        return last_output_;
      }
      T filtered_value = alpha_.cwiseProduct(this->NanHandle(input)) +
                         one_minus_alpha_.cwiseProduct(last_output_);
      last_output_ = filtered_value;
      return filtered_value.cwiseMin(upper_bound_).cwiseMax(lower_bound_);
    } else {
      if (!init_flag_) [[unlikely]] {
        init_flag_ = true;
        last_output_ = this->NanHandle(input);
        return std::max(lower_bound_, std::min(upper_bound_, last_output_));
      }
      T filtered_value =
          alpha_ * this->NanHandle(input) + one_minus_alpha_ * last_output_;
      last_output_ = filtered_value;
      return std::max(lower_bound_, std::min(upper_bound_, filtered_value));
    }
  }

  virtual void Reset() final {
    init_flag_ = false;
    if constexpr (is_eigen_vector_v<T>) {
      this->last_input_.setZero();
    } else {
      this->last_input_ = 0;
    }
  };

 private:
  T lower_bound_;
  T upper_bound_;
  T alpha_;
  T one_minus_alpha_;

  bool init_flag_;
  T last_output_;
};
}  // namespace ovinf

#endif  // !FILTER_LOW_PASS_HPP
