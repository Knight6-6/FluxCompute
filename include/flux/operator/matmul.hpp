#pragma once

#include <flux/tensor/tensor.hpp>
#include <flux/backend/cpu/matmul.hpp>
#include <cstddef>
#include <stdexcept>

namespace flux::ops {

// 矩阵乘：C = A * B
//
//     A 形状 (M, K)  ·  B 形状 (K, N)  ->  C 形状 (M, N)
//
// 只支持二维。批处理（batched matmul）、广播、转置参数都**没做**——
// 这是第一版，先把最核心的那个语义定下来。需要的话应当作为独立算子另加，
// 而不是给 matmul 堆参数。
//
// 与现有算子的结构差异：既不套 outer/axis/inner（那是"沿某一维归约"的
// 模式），也不像逐元素算子那样输出与输入同形。它是第一个输出形状由两个
// 输入的**共同**约束决定的算子。
//
// 优化状态：内核是朴素三重循环（仅选了非病态的循环序 i-k-j，见
// backend/cpu/matmul.hpp）。分块 / SIMD / 多线程都没做——按 README 原则 #4，
// 那类优化要有基准支撑，先有正确版本与基准数字。
template <typename T>
tensor::Tensor<T> matmul(const tensor::Tensor<T>& a, const tensor::Tensor<T>& b) {
    const auto& ad = a.shape().dimensions();
    const auto& bd = b.shape().dimensions();

    if (ad.size() != 2 || bd.size() != 2) {
        throw std::invalid_argument("matmul: both operands must be 2-D");
    }

    const std::size_t m  = ad[0];
    const std::size_t k  = ad[1];
    const std::size_t k2 = bd[0];
    const std::size_t n  = bd[1];

    if (k != k2) {
        throw std::invalid_argument("matmul: inner dimensions do not match");
    }

    tensor::Tensor<T> c{tensor::Shape({m, n})};
    backend::cpu::matmul_kernel(a.data(), b.data(), c.data(), m, k, n);
    return c;
}

}
