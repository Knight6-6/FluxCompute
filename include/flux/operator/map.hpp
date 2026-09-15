#pragma once

#include <flux/tensor/tensor.hpp>
#include <flux/backend/cpu/elementwise.hpp>
#include <cstddef>
#include <stdexcept>

namespace flux::ops {

namespace detail {

// N 元融合要求所有输入形状一致——与 BinaryElementwise 的约定相同，不支持广播。
// 非模板自由函数在 header-only 库里必须 inline，否则跨翻译单元 ODR 冲突。
inline void require_same_shape(const tensor::Shape& a, const tensor::Shape& b) {
    if (a != b) {
        throw std::invalid_argument("Shapes of tensors must match for fused elementwise operation");
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// 融合的逐元素算子：把多个张量在**一趟遍历**里合并成输出。
//
//     auto r = ops::map3(a, b, c, [](float x, float y, float z) {
//         return (x + y) * z;
//     });
//
// 等价的分离写法 ops::mul(ops::add(a, b), c) 要做两趟遍历、产生一个中间张量。
//
// 为什么值得：逐元素算子本身是**内存受限**的（实测多核并行只有 0.92x，
// 见 docs/04-计算图与运行时.md），真正的杠杆是减少内存流量而不是提高算力。
// 融合把 N 趟遍历压成 1 趟：
//
//     实测 1M float，(a+b)*c     两趟 0.574 ms -> 一趟 0.244 ms   2.35x
//     实测 1M float，(a+b)*c-a   三趟 0.935 ms -> 一趟 0.242 ms   3.86x
//
// 收益随链长增长，因为省掉的遍历次数随链长线性增加。
//
// 两点约定：
//   - **不跳过 NaN**。与 reduce / scan 同理：lambda 是任意标量函数，
//     库无从知道它该怎么处理 NaN，故一律照常参与。
//   - **形状必须完全一致**，不做广播。
//
// 关于"自动融合"：计算图里的算子是类型擦除的 std::function，图看不出一个
// 节点是不是逐元素、也无从取出它的标量函数，所以自动融合在运行时图这个
// 设计下做不到（真做要按元素做类型擦除的间接调用，开销可能超过省下的流量）。
// 因此这里提供的是**能力**：由上层在绑定算子时把整条链写成一个融合表达式。
// 见 docs/03-算子与内核.md。
// ---------------------------------------------------------------------------

template <typename T, typename F>
tensor::Tensor<T> map2(const tensor::Tensor<T>& a,
                       const tensor::Tensor<T>& b,
                       F f) {
    detail::require_same_shape(a.shape(), b.shape());

    tensor::Tensor<T> result(a.shape());
    backend::cpu::zip2(a.data(), b.data(), result.data(), a.numel(), f);
    return result;
}

template <typename T, typename F>
tensor::Tensor<T> map3(const tensor::Tensor<T>& a,
                       const tensor::Tensor<T>& b,
                       const tensor::Tensor<T>& c,
                       F f) {
    detail::require_same_shape(a.shape(), b.shape());
    detail::require_same_shape(a.shape(), c.shape());

    tensor::Tensor<T> result(a.shape());
    backend::cpu::zip3(a.data(), b.data(), c.data(), result.data(), a.numel(), f);
    return result;
}

template <typename T, typename F>
tensor::Tensor<T> map4(const tensor::Tensor<T>& a,
                       const tensor::Tensor<T>& b,
                       const tensor::Tensor<T>& c,
                       const tensor::Tensor<T>& d,
                       F f) {
    detail::require_same_shape(a.shape(), b.shape());
    detail::require_same_shape(a.shape(), c.shape());
    detail::require_same_shape(a.shape(), d.shape());

    tensor::Tensor<T> result(a.shape());
    backend::cpu::zip4(a.data(), b.data(), c.data(), d.data(), result.data(), a.numel(), f);
    return result;
}

}
