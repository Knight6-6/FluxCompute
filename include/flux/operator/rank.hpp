#pragma once

#include <flux/tensor/tensor.hpp>
#include <flux/backend/cpu/rank.hpp>
#include <flux/operator/axis_util.hpp>

namespace flux::ops {

// 名次（pandas method='average'：并列取平均名次），NaN 保持 NaN。
// 返回 Tensor<T>：float 精确；整数 T 的并列小数被截断（如 2.5 -> 2）。
template <typename T>
class Rank {
public:
    tensor::Tensor<T> operator()(const tensor::Tensor<T>& input,
                                 std::size_t axis = 0,
                                 bool pct = false) const {
        const auto info = detail::compute_axis_info(input.shape(), axis);
        tensor::Tensor<T> result(input.shape());
        backend::cpu::rank_axis(input.data(), result.data(),
                                info.outer_size, info.axis_size, info.inner_size,
                                pct);
        return result;
    }
};

template <typename T>
tensor::Tensor<T> rank(const tensor::Tensor<T>& input,
                       std::size_t axis = 0,
                       bool pct = false) {
    return Rank<T>{}(input, axis, pct);
}

}
