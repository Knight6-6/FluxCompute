#pragma once

#include <flux/tensor/tensor.hpp>
#include <flux/backend/cpu/scan.hpp>
#include <flux/operator/axis_util.hpp>
#include <cstddef>

namespace flux::ops {

// 前缀扫描：把调用方给的二元算子沿轴做左折叠，输出每一步的累加值。
//
// 与 reduce 共用同一套约定（算子即 callable、单位元显式、不跳过 NaN），
// 详见 backend/cpu/reduction.hpp。两点与 reduce 的差别：
//
//   - **输出形状 == 输入形状**（reduce 会去掉被折叠的那一维）；
//   - **只提供按轴版本**。归约能折叠成标量所以有一个"全量"重载，而"整张量
//     的前缀扫描"对 N 维没有自然含义（要先拍平成什么顺序？），故不给。
//
// axis 放在最后并默认 0，与 shift / rolling_mean / sort / rank 的既有约定一致。
//
// 用法：
//
//     // 沿 axis 0 累加（1-D 即 cumsum）
//     tensor::Tensor<float> cs = ops::scan(t, std::plus<float>{}, 0.f);
//
//     // 沿 axis 1 取累计最大
//     tensor::Tensor<float> cm = ops::scan(
//         t, [](float a, float b) { return a > b ? a : b; },
//         std::numeric_limits<float>::lowest(), /*axis=*/1);

template <typename T, typename BinaryOp>
tensor::Tensor<T> scan(const tensor::Tensor<T>& input,
                       BinaryOp op,
                       T identity,
                       std::size_t axis = 0) {
    const auto info = detail::compute_axis_info(input.shape(), axis);

    tensor::Tensor<T> result(input.shape());

    backend::cpu::scan_axis(input.data(), result.data(),
                            info.outer_size, info.axis_size, info.inner_size,
                            op, identity);

    return result;
}

}
