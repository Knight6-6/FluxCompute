#pragma once

#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>

namespace flux::backend::cpu {

// 沿轴稳定排序，NaN 恒排最后（升/降序一致，pandas na_position='last'）。
// 比较器必须是严格弱序：两个 NaN 分支先于数值比较，使 NaN 互相等价且为"最大值"；
// 裸 a < b 会让 NaN 与所有元素等价，破坏 stable_sort。
// 整数 T：std::isnan(int) 恒 false，比较器退化为普通 < / >。
template <typename T>
void sort_axis(const T* input, T* output,
               std::size_t outer_size,
               std::size_t axis_size,
               std::size_t inner_size,
               bool ascending) {

    // 临时缓冲区在循环外分配一次并复用，避免每个 (o,i) 切片都触发堆分配
    std::vector<T> temp(axis_size);

    for (std::size_t o = 0; o < outer_size; ++o) {
        for (std::size_t i = 0; i < inner_size; ++i) {
            for (std::size_t a = 0; a < axis_size; ++a) {
                temp[a] = input[(o * axis_size + a) * inner_size + i];
            }

            std::stable_sort(temp.begin(), temp.end(),
                [ascending](const T& a, const T& b) {
                    if (std::isnan(a)) return false;   // NaN 不排在任何元素前
                    if (std::isnan(b)) return true;    // 但排在所有非 NaN 后
                    return ascending ? (a < b) : (a > b);
                });

            for (std::size_t a = 0; a < axis_size; ++a) {
                output[(o * axis_size + a) * inner_size + i] = temp[a];
            }
        }
    }
}

}
