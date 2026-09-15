#pragma once

#include <flux/tensor/tensor.hpp>
#include <flux/backend/cpu/sort.hpp>
#include <flux/operator/axis_util.hpp>

namespace flux::ops {

// 沿轴稳定排序，NaN 排最后，升/降序可选。只返回排序后的值（argsort 后续再加）。
template <typename T>
class Sort {
public:
    tensor::Tensor<T> operator()(const tensor::Tensor<T>& input,
                                 std::size_t axis = 0,
                                 bool ascending = true) const {
        const auto info = detail::compute_axis_info(input.shape(), axis);
        tensor::Tensor<T> result(input.shape());
        backend::cpu::sort_axis(input.data(), result.data(),
                                info.outer_size, info.axis_size, info.inner_size,
                                ascending);
        return result;
    }
};

template <typename T>
tensor::Tensor<T> sort(const tensor::Tensor<T>& input,
                       std::size_t axis = 0,
                       bool ascending = true) {
    return Sort<T>{}(input, axis, ascending);
}

}
