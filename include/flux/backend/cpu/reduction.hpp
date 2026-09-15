#pragma once
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace flux::backend::cpu {

// 求和（pandas 风格 skipna：NaN 不参与求和）。
// 无有效值时返回 0，即加法单位元 —— 这点与 mean/max/min 的"输出 NaN"不同：
// sum 的空集语义本就是 0（pandas 对全 NaN 切片求和同样为 0），
// 因此无需额外计数，total 保持初值即可。
template <typename T>
T sum_all(const T* input, std::size_t size) {
    T total{};
    #pragma omp simd reduction(+:total)
    for (std::size_t i = 0; i < size; ++i) {
        const T v = input[i];
        if (!std::isnan(v)) total += v;
    }
    return total;
}

template <typename T>
void sum_axis(const T* input, T* output, std::size_t outer_size, std::size_t axis_size,  std::size_t inner_size) {
    
    std::fill_n(output, outer_size * inner_size, T{0});

    for (std::size_t o = 0; o < outer_size; ++o) {
        for (std::size_t a = 0; a < axis_size; ++a) {
            const T* in_ptr = input + (o * axis_size + a) * inner_size;
            T* out_ptr = output + o * inner_size;
            
            #pragma omp simd
            for (std::size_t i = 0; i < inner_size; ++i) {
                const T v = in_ptr[i];
                if (!std::isnan(v)) out_ptr[i] += v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mean / max / min：pandas 风格 skipna —— NaN 不参与统计，整个切片无有效值时输出 NaN。
// 整数 T：std::isnan(int) 恒 false，skipna 分支永不进入，退化为普通归约。
// ---------------------------------------------------------------------------

template <typename T>
T mean_all(const T* input, std::size_t size) {
    T sum{};
    std::size_t valid = 0;

    #pragma omp simd reduction(+:sum,valid)
    for (std::size_t i = 0; i < size; ++i) {
        const T v = input[i];
        if (!std::isnan(v)) {
            sum += v;
            ++valid;
        }
    }

    // 整数 T 为整除截断，与现有算术语义一致
    return valid == 0 ? std::numeric_limits<T>::quiet_NaN()
                      : sum / static_cast<T>(valid);
}

template <typename T>
void mean_axis(const T* input, T* output,
               std::size_t outer_size,
               std::size_t axis_size,
               std::size_t inner_size) {

    // 循环序与 rolling_mean_axis 一致：外层 o/i 定位切片，内层沿 axis 走，
    // 每个切片的 sum/valid 是标量，不需要额外的 per-slice 缓冲。
    for (std::size_t o = 0; o < outer_size; ++o) {
        for (std::size_t i = 0; i < inner_size; ++i) {
            const std::size_t base = o * axis_size * inner_size + i;

            T sum{};
            std::size_t valid = 0;

            for (std::size_t a = 0; a < axis_size; ++a) {
                const T v = input[base + a * inner_size];
                if (!std::isnan(v)) {
                    sum += v;
                    ++valid;
                }
            }

            output[o * inner_size + i] = valid == 0
                ? std::numeric_limits<T>::quiet_NaN()
                : sum / static_cast<T>(valid);
        }
    }
}

// max / min 共用一套骨架，只差一个比较符，不必写两遍。
namespace detail {

// 极值单位元：浮点取 ±inf，整数取 lowest()/max()。
// 不可用 T{}：整数 T 没有 NaN，全负切片若用 0 兜底会得到错误结果。
template <typename T>
constexpr T max_identity() {
    return std::is_floating_point<T>::value ? -std::numeric_limits<T>::infinity()
                                            : std::numeric_limits<T>::lowest();
}

template <typename T>
constexpr T min_identity() {
    return std::is_floating_point<T>::value ? std::numeric_limits<T>::infinity()
                                            : std::numeric_limits<T>::max();
}

// Compare 决定取向（std::greater -> max，std::less -> min）。
// 刻意不加 simd 提示：带 NaN 过滤的条件极值不是 OpenMP 规约子句能直接表达的形式，
// 硬套 pragma 等于对编译器做无依据的承诺。按"先正确、再优化"，
// 待 benchmark 确认它是瓶颈后，再单独设计向量化路径。
template <typename T, typename Compare>
T extremum_all(const T* input, std::size_t size, T identity) {
    T best = identity;
    std::size_t valid = 0;

    for (std::size_t i = 0; i < size; ++i) {
        const T v = input[i];
        if (!std::isnan(v)) {
            ++valid;
            if (Compare{}(v, best)) best = v;
        }
    }

    return valid == 0 ? std::numeric_limits<T>::quiet_NaN() : best;
}

template <typename T, typename Compare>
void extremum_axis(const T* input, T* output,
                   std::size_t outer_size,
                   std::size_t axis_size,
                   std::size_t inner_size,
                   T identity) {

    for (std::size_t o = 0; o < outer_size; ++o) {
        for (std::size_t i = 0; i < inner_size; ++i) {
            const std::size_t base = o * axis_size * inner_size + i;

            T best = identity;
            std::size_t valid = 0;

            for (std::size_t a = 0; a < axis_size; ++a) {
                const T v = input[base + a * inner_size];
                if (!std::isnan(v)) {
                    ++valid;
                    if (Compare{}(v, best)) best = v;
                }
            }

            output[o * inner_size + i] = valid == 0
                ? std::numeric_limits<T>::quiet_NaN()
                : best;
        }
    }
}

} // namespace detail

template <typename T>
T max_all(const T* input, std::size_t size) {
    return detail::extremum_all<T, std::greater<T>>(
        input, size, detail::max_identity<T>());
}

template <typename T>
void max_axis(const T* input, T* output,
              std::size_t outer_size,
              std::size_t axis_size,
              std::size_t inner_size) {
    detail::extremum_axis<T, std::greater<T>>(
        input, output, outer_size, axis_size, inner_size, detail::max_identity<T>());
}

template <typename T>
T min_all(const T* input, std::size_t size) {
    return detail::extremum_all<T, std::less<T>>(
        input, size, detail::min_identity<T>());
}

template <typename T>
void min_axis(const T* input, T* output,
              std::size_t outer_size,
              std::size_t axis_size,
              std::size_t inner_size) {
    detail::extremum_axis<T, std::less<T>>(
        input, output, outer_size, axis_size, inner_size, detail::min_identity<T>());
}

}