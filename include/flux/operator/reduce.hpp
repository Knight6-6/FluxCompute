#pragma once

#include <flux/tensor/tensor.hpp>
#include <flux/backend/cpu/reduction.hpp>
#include <flux/operator/axis_util.hpp>
#include <cstddef>

namespace flux::ops {

// 通用归约：把调用方给的二元算子沿轴（或全量）左折叠。
//
// 两个必须知道的约定（理由见 backend/cpu/reduction.hpp 的详细说明）：
//   - **不跳过 NaN**。sum/mean/max/min 有 skipna 语义是因为它们知道 NaN
//     对各自运算意味着什么；通用归约拿到的是任意算子，无从判断，故一律照常参与。
//   - **单位元必须显式给出**。它在语义上是折叠的起点，不是可有可无的参数：
//     求和的单位元是 0，求最大值的是 lowest()，用 0 兜底会让全负切片取到错误的 0。
//
// 用法：
//
//     float total = ops::reduce(t, std::plus<float>{}, 0.f);
//
//     tensor::Tensor<float> col_max = ops::reduce(
//         t, /*axis=*/0,
//         [](float a, float b) { return a > b ? a : b; },
//         std::numeric_limits<float>::lowest());
//
// 这里只提供自由函数，不像 Sum/Mean 那样再包一个类：算子是调用方的参数
// 而非类的模板实参，包一层类只会让调用变啰嗦。

template <typename T, typename BinaryOp>
T reduce(const tensor::Tensor<T>& input, BinaryOp op, T identity) {
    return backend::cpu::reduce_all(input.data(), input.numel(), op, identity);
}

template <typename T, typename BinaryOp>
tensor::Tensor<T> reduce(const tensor::Tensor<T>& input,
                         std::size_t axis,
                         BinaryOp op,
                         T identity) {
    const auto info = detail::compute_axis_info(input.shape(), axis);

    tensor::Tensor<T> result{tensor::Shape(info.out_dims)};

    backend::cpu::reduce_axis(input.data(), result.data(),
                              info.outer_size, info.axis_size, info.inner_size,
                              op, identity);

    return result;
}

}
