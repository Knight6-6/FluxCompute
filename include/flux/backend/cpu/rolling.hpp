#pragma once
#include <cstddef>
#include <limits>
#include <cmath>
#include <type_traits>

namespace flux::backend::cpu {

template <typename T>
void rolling_mean_axis(const T* input, T* output,
                        std::size_t outer_size,
                        std::size_t axis_size,
                        std::size_t inner_size,
                        std::size_t window,
                        std::size_t min_periods,
                        T fill_value) {
    
    for (std::size_t o = 0; o < outer_size; ++o) {
        for (std::size_t i = 0; i < inner_size; ++i) {

            const std::size_t stride = inner_size;
            const std::size_t base_offset = o * axis_size * inner_size + i;

            T current_sum = T{0};
            std::size_t valid_count = 0;

            for (std::size_t a = 0; a < axis_size; ++a) {
                const T new_val = input[base_offset + a * stride];
                
                if (!std::isnan(new_val)) {
                    current_sum += new_val;
                    valid_count++;
                }

                if (a >= window) {
                    const T old_val = input[base_offset + (a - window) * stride];
                    if (!std::isnan(old_val)) {
                        current_sum -= old_val;
                        valid_count--;
                    }
                }

                if (valid_count >= min_periods && valid_count > 0) {
                    output[base_offset + a * stride] = current_sum / static_cast<T>(valid_count);
                } else {
                    output[base_offset + a * stride] = fill_value;
                }
            }
        }
    }
}

// 滚动样本标准差。与 rolling_mean_axis 同一套约定：按轴、增量滑窗 O(n)、
// skipna、min_periods 控制最小有效数、不足则填 fill_value。
//
// ---------------------------- 数值上的一处关键取舍 ----------------------------
// 累加器统一提升到 double（long double 除外）：
//
//     using Acc = std::conditional_t<std::is_same<T, long double>::value,
//                                    long double, double>;
//
// 原因不是洁癖，是实测出来的。方差用 sum_sq - sum²/n 计算存在抵消风险，
// 用 float 累加时在真实量级下直接失真（对照"先减均值"的稳定两趟参照）：
//     收益率 ~0.01 量级      相对误差 3.0e-07   —— 尚可
//     价格 ~100 量级         相对误差 3.4e-02   —— 不可用
//     价格 ~100 且波动极小    相对误差 1.6e+03   —— 完全失真
//     成交额 ~1e6 量级       相对误差 1.0e+00   —— 等于没算
// 提升到 double 后，在 float 能表示的范围内误差降到 ~1e-10 或更好。
//
// 残留局限：均值² / 方差 超过 ~1e7 时 double 也会开始丢有效位。要覆盖那种
// 量级得换 Welford 递推或分块两趟——但那种数据的 mean/std 比已经超出 float
// 的表示能力，属于人造输入，暂不处理。若将来真有需求，这里是明确的改造点。
// ---------------------------------------------------------------------------
//
// 注意样本数要同时满足 min_periods 与 ddof：方差要除以 n - ddof，
// n <= ddof 时无定义（ddof 默认 1，即至少要有 2 个有效样本）。
template <typename T>
void rolling_std_axis(const T* input, T* output,
                      std::size_t outer_size,
                      std::size_t axis_size,
                      std::size_t inner_size,
                      std::size_t window,
                      std::size_t min_periods,
                      T fill_value,
                      std::size_t ddof) {

    using Acc = std::conditional_t<std::is_same<T, long double>::value, long double, double>;

    for (std::size_t o = 0; o < outer_size; ++o) {
        for (std::size_t i = 0; i < inner_size; ++i) {

            const std::size_t stride = inner_size;
            const std::size_t base_offset = o * axis_size * inner_size + i;

            Acc sum = Acc{0};
            Acc sum_sq = Acc{0};
            std::size_t valid_count = 0;

            for (std::size_t a = 0; a < axis_size; ++a) {
                const T new_val = input[base_offset + a * stride];

                if (!std::isnan(new_val)) {
                    const Acc v = static_cast<Acc>(new_val);
                    sum += v;
                    sum_sq += v * v;
                    valid_count++;
                }

                if (a >= window) {
                    const T old_val = input[base_offset + (a - window) * stride];
                    if (!std::isnan(old_val)) {
                        const Acc v = static_cast<Acc>(old_val);
                        sum -= v;
                        sum_sq -= v * v;
                        valid_count--;
                    }
                }

                if (valid_count >= min_periods && valid_count > ddof) {
                    const Acc n = static_cast<Acc>(valid_count);
                    Acc var = (sum_sq - sum * sum / n) / (n - static_cast<Acc>(ddof));
                    // 抵消可能让方差轻微为负，开方前夹到 0，否则会得到 NaN
                    if (var < Acc{0}) var = Acc{0};
                    output[base_offset + a * stride] = static_cast<T>(std::sqrt(var));
                } else {
                    output[base_offset + a * stride] = fill_value;
                }
            }
        }
    }
}

}