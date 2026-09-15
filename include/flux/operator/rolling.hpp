#pragma once
#include <flux/tensor/tensor.hpp>
#include <flux/backend/cpu/rolling.hpp>
#include <flux/operator/axis_util.hpp>
#include <cmath>
#include <limits>

namespace flux::ops {

template <typename T>
class RollingMean {
public:
    tensor::Tensor<T> operator()(const tensor::Tensor<T>& input,
                                 std::size_t window,
                                 std::size_t min_periods = 0,
                                 std::size_t axis = 0,
                                 T fill_value = std::numeric_limits<T>::quiet_NaN()) const {
        
        if (window == 0) {
            throw std::invalid_argument("Window size must be greater than 0");
        }

        // 默认 min_periods 等于 window 大小
        if (min_periods == 0) {
            min_periods = window;
        }

        const auto info = detail::compute_axis_info(input.shape(), axis);

        tensor::Tensor<T> result(input.shape());

        backend::cpu::rolling_mean_axis(
            input.data(), result.data(),
            info.outer_size, info.axis_size, info.inner_size,
            window, min_periods, fill_value
        );

        return result;
    }
};

template <typename T>
tensor::Tensor<T> rolling_mean(const tensor::Tensor<T>& input,
                               std::size_t window,
                               std::size_t min_periods = 0,
                               std::size_t axis = 0,
                               T fill_value = std::numeric_limits<T>::quiet_NaN()) {
    return RollingMean<T>{}(input, window, min_periods, axis, fill_value);
}

}