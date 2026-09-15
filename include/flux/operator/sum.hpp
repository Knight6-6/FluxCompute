#pragma once
#include <flux/tensor/tensor.hpp>
#include <flux/backend/cpu/reduction.hpp>
#include <flux/operator/axis_util.hpp>

namespace flux::ops {

// 求和（pandas 风格 skipna：跳过 NaN）。
// 与 mean/max/min 的差别：全 NaN 切片求和为 0，而非 NaN。
template <typename T>
class Sum {
public:
    T operator()(const tensor::Tensor<T>& input) const {
        return backend::cpu::sum_all(input.data(), input.numel());
    }

    tensor::Tensor<T> operator()(const tensor::Tensor<T>& input, std::size_t axis) const {
        const auto info = detail::compute_axis_info(input.shape(), axis);
        tensor::Tensor<T> result{tensor::Shape(info.out_dims)};
        backend::cpu::sum_axis(input.data(), result.data(),
                               info.outer_size, info.axis_size, info.inner_size);
        return result;
    }
};

template <typename T>
T sum(const tensor::Tensor<T>& input) {
    return Sum<T>{}(input);
}

template <typename T>
tensor::Tensor<T> sum(const tensor::Tensor<T>& input, std::size_t axis) {
    return Sum<T>{}(input, axis);
}

}