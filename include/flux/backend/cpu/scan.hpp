#pragma once
#include <cstddef>

namespace flux::backend::cpu {

// 前缀扫描：左折叠的逐前缀结果。与 reduce 是同一套骨架，区别只在于
// reduce 只留最后一个累加值，scan 把每一步的累加值都写出去。
//
// 约定与 reduce 完全一致（详见 reduction.hpp 中通用归约的说明）：
//   - 二元算子与单位元都由调用方给出，单位元不做兜底；
//   - 不跳过 NaN——任意二元算子无从知道 NaN 该被怎么处理；
//   - 折叠顺序固定从左到右，对非交换算子结果确定。
//
// 注意后两条意味着它与 pandas cumsum 不同：pandas 在 NaN 处输出 NaN 但
// 继续累加（[1, NaN, 3] -> [1, NaN, 4]），这里一旦吃到 NaN 之后全是 NaN
// （-> [1, NaN, NaN]）。需要 pandas 语义的话应当在算子层显式处理 NaN，
// 而不是把这个假设埋进通用内核。
//
// 不加 simd 提示：前缀扫描天然存在循环携带依赖，向量化需要专门的
// 分段扫描算法，属于"先正确、再优化"里后一半的事。
template <typename T, typename BinaryOp>
void scan_axis(const T* input, T* output,
               std::size_t outer_size,
               std::size_t axis_size,
               std::size_t inner_size,
               BinaryOp op, T identity) {

    // 循环序 o/i/a：每个切片的前缀累加器是标量，不需要 per-slice 缓冲。
    for (std::size_t o = 0; o < outer_size; ++o) {
        for (std::size_t i = 0; i < inner_size; ++i) {
            const std::size_t base = o * axis_size * inner_size + i;

            T acc = identity;
            for (std::size_t a = 0; a < axis_size; ++a) {
                const std::size_t idx = base + a * inner_size;
                // 先读 input 再写 output，且后续只依赖 acc 与 input——
                // 因此 input 与 output 允许是同一块内存。
                acc = op(acc, input[idx]);
                output[idx] = acc;
            }
        }
    }
}

}
