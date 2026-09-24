#pragma once

#include <cstddef>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <type_traits>

namespace flux::backend::cpu {

namespace detail {

template <typename U>
inline bool is_nan_value(const U& val) {
    if constexpr (std::is_floating_point_v<U>) {
        return std::isnan(val);
    } else {
        return false;
    }
}

template <typename T>
struct IndexedElement {
    T val;
    std::size_t orig_a;

    bool operator<(const IndexedElement& other) const noexcept {
        return val < other.val;
    }
};

} // namespace detail

// pandas method='average'：并列取平均名次；na_option='keep'：NaN 输出 NaN。
// 基于排序的 O(N log N) 算法，复用切片缓冲区避免堆分配。
// 整数 T：std::isnan(int) 恒 false，NaN 分支永不进入；并列小数被 static_cast<T> 截断。
template <typename T>
void rank_axis(const T* input, T* output,
               std::size_t outer_size,
               std::size_t axis_size,
               std::size_t inner_size,
               bool pct) {

    if (axis_size == 0) return;

    // 预分配缓冲区并在切片间复用，避免每次循环触发堆分配
    std::vector<detail::IndexedElement<T>> elements;
    elements.reserve(axis_size);

    for (std::size_t o = 0; o < outer_size; ++o) {
        for (std::size_t i = 0; i < inner_size; ++i) {
            elements.clear();

            // 1. 过滤 NaN，收集非 NaN 元素及其原始轴向坐标
            for (std::size_t a = 0; a < axis_size; ++a) {
                const std::size_t idx = (o * axis_size + a) * inner_size + i;
                const T v = input[idx];

                if (detail::is_nan_value(v)) {
                    output[idx] = std::numeric_limits<T>::quiet_NaN();
                } else {
                    elements.push_back({v, a});
                }
            }

            const std::size_t valid = elements.size();
            if (valid == 0) {
                continue;
            }

            // 2. 对有效元素排序 (O(N log N))
            std::sort(elements.begin(), elements.end());

            // 3. 统计并列 (ties) 并计算平均名次 (method='average')
            std::size_t start = 0;
            while (start < valid) {
                std::size_t end = start + 1;
                while (end < valid && elements[end].val == elements[start].val) {
                    ++end;
                }

                // 名次区间为 [start + 1, end]，平均名次为 ((start + 1) + end) / 2.0
                double avg_rank = (static_cast<double>(start + 1) + static_cast<double>(end)) * 0.5;
                if (pct) {
                    avg_rank /= static_cast<double>(valid);
                }

                const T rank_val = static_cast<T>(avg_rank);
                for (std::size_t k = start; k < end; ++k) {
                    const std::size_t out_idx = (o * axis_size + elements[k].orig_a) * inner_size + i;
                    output[out_idx] = rank_val;
                }

                start = end;
            }
        }
    }
}

} // namespace flux::backend::cpu
