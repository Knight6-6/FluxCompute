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

// 滚动标准差（pandas 风格 skipna，与 rolling_mean 同一套语义）。
//
// 前四个参数与 rolling_mean 位置完全一致，便于在两者之间切换；ddof 追加在
// 末尾（自由度，默认 1 即样本标准差，与 pandas 的 rolling().std() 一致）。
//
// 数值上有一处刻意取舍：累加器统一提升到 double。用 float 累加 sum/sumsq
// 在价格、成交额这种真实量级下会因抵消而严重失真（实测成交额 ~1e6 量级
// 相对误差 1.00e+00）——详见 backend/cpu/rolling.hpp 里的实测数据。
template <typename T>
class RollingStd {
public:
    tensor::Tensor<T> operator()(const tensor::Tensor<T>& input,
                                 std::size_t window,
                                 std::size_t min_periods = 0,
                                 std::size_t axis = 0,
                                 T fill_value = std::numeric_limits<T>::quiet_NaN(),
                                 std::size_t ddof = 1) const {

        if (window == 0) {
            throw std::invalid_argument("Window size must be greater than 0");
        }

        // 默认 min_periods 等于 window 大小
        if (min_periods == 0) {
            min_periods = window;
        }

        const auto info = detail::compute_axis_info(input.shape(), axis);

        tensor::Tensor<T> result(input.shape());

        backend::cpu::rolling_std_axis(
            input.data(), result.data(),
            info.outer_size, info.axis_size, info.inner_size,
            window, min_periods, fill_value, ddof
        );

        return result;
    }
};

template <typename T>
tensor::Tensor<T> rolling_std(const tensor::Tensor<T>& input,
                              std::size_t window,
                              std::size_t min_periods = 0,
                              std::size_t axis = 0,
                              T fill_value = std::numeric_limits<T>::quiet_NaN(),
                              std::size_t ddof = 1) {
    return RollingStd<T>{}(input, window, min_periods, axis, fill_value, ddof);
}

}