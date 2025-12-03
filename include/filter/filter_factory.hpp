#ifndef FILTER_FACTORY_HPP
#define FILTER_FACTORY_HPP

#include "filter/filter_base.hpp"
#include "filter/filter_low_pass.hpp"
#include "filter/filter_mean.hpp"
#include "filter/filter_pass_through.hpp"

namespace ovinf {

class FilterFactory {
 public:
  template <typename T = Eigen::Matrix<float, Eigen::Dynamic, 1>>
  static std::shared_ptr<FilterBase<T>> CreateFilter(YAML::Node const &config) {
    std::string filter_type = config["type"].as<std::string>();
    if (filter_type == "None") {
      return std::make_shared<FilterBase<T>>(config);
    }
    if (filter_type == "Mean") {
      return std::make_shared<MeanFilter<T>>(config);
    } else if (filter_type == "PassThrough") {
      return std::make_shared<PassThroughFilter<T>>(config);
    } else if (filter_type == "LowPass") {
      return std::make_shared<LowPassFilter<T>>(config);
    } else {
      throw std::invalid_argument("Unknown filter type: " + filter_type);
    }
  }
};

}  // namespace ovinf

#endif  // !FILTER_FACTORY_HPP
