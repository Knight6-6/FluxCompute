#pragma once

#include <cstddef>
#include <cmath>
#include <limits>

namespace flux::backend::cpu {

// pandas method='average'：并列取平均名次；na_option='keep'：NaN 输出 NaN。
// 计数法，每切片 O(N²)，正确性优先；后续可换基于排序的 O(N log N)。
// 整数 T：std::isnan(int) 恒 false，NaN 分支永不进入；并列小数被 static_cast<T> 截断。
template <typename T>
void rank_axis(const T* input, T* output,
               std::size_t outer_size,
               std::size_t axis_size,
               std::size_t inner_size,
               bool pct) {

    for (std::size_t o = 0; o < outer_size; ++o) {
        for (std::size_t i = 0; i < inner_size; ++i) {

            // 切片内有效（非 NaN）个数，供 pct 归一化
            std::size_t valid = 0;
            for (std::size_t a = 0; a < axis_size; ++a) {
                if (!std::isnan(input[(o * axis_size + a) * inner_size + i])) ++valid;
            }

            for (std::size_t a = 0; a < axis_size; ++a) {
                const std::size_t idx = (o * axis_size + a) * inner_size + i;
                const T v = input[idx];

                if (std::isnan(v)) {  // 整数 T 永不进入此分支
                    output[idx] = std::numeric_limits<T>::quiet_NaN();
                    continue;
                }

                std::size_t less = 0, equal = 0;
                for (std::size_t b = 0; b < axis_size; ++b) {
                    const T w = input[(o * axis_size + b) * inner_size + i];
                    if (std::isnan(w)) continue;
                    if (w < v) ++less;
                    else if (w == v) ++equal;  // 含自身，equal >= 1
                }

                // 平均名次 = 1 + 比 v 小的个数 + (与 v 相等的个数 - 1) / 2
                double rank_val = 1.0 + static_cast<double>(less)
                                + (static_cast<double>(equal) - 1.0) / 2.0;
                if (pct) rank_val /= static_cast<double>(valid);  // 本分支保证 valid >= 1
                output[idx] = static_cast<T>(rank_val);
            }
        }
    }
}

}
