#pragma once

#include <flux/tensor/tensor.hpp>
#include <flux/backend/cpu/reduction.hpp>
#include <flux/operator/axis_util.hpp>

namespace flux::ops {

// 最小值归约（pandas 风格 skipna：跳过 NaN，全 NaN 切片输出 NaN）。
template <typename T>
class Min {
public:
    T operator()(const tensor::Tensor<T>& input) const {
        return backend::cpu::min_all(input.data(), input.numel());
    }

    tensor::Tensor<T> operator()(const tensor::Tensor<T>& input, std::size_t axis) const {
        const auto info = detail::compute_axis_info(input.shape(), axis);
        tensor::Tensor<T> result{tensor::Shape(info.out_dims)};
        backend::cpu::min_axis(input.data(), result.data(),
                               info.outer_size, info.axis_size, info.inner_size);
        return result;
    }
};

template <typename T>
T min(const tensor::Tensor<T>& input) {
    return Min<T>{}(input);
}

template <typename T>
tensor::Tensor<T> min(const tensor::Tensor<T>& input, std::size_t axis) {
    return Min<T>{}(input, axis);
}

}
