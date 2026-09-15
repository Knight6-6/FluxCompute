#pragma once

#include <flux/tensor/shape.hpp>
#include <cstddef>
#include <vector>
#include <stdexcept>

// 共享的"按轴分解"工具：把多维张量按 axis 拆成 outer/axis/inner 三段循环，
// 供归约、shift、rolling 等算子复用（原本在 sum/shift/rolling 里各自重复实现）。
namespace flux::ops::detail {

struct AxisInfo {
    std::size_t outer_size = 1;  // axis 之前各维的乘积
    std::size_t axis_size  = 0;  // 被操作的那一维大小
    std::size_t inner_size = 1;  // axis 之后各维的乘积
    std::vector<std::size_t> out_dims;  // 去掉 axis 之后的输出形状（空则回退 {1}）
};

// 非模板自由函数必须 inline，否则 header-only 库跨翻译单元会 ODR 冲突。
inline AxisInfo compute_axis_info(const tensor::Shape& shape, std::size_t axis) {
    const auto& dims = shape.dimensions();
    if (axis >= dims.size()) {
        throw std::out_of_range("Axis out of bounds");
    }

    AxisInfo info;
    for (std::size_t i = 0; i < axis; ++i)                info.outer_size *= dims[i];
    info.axis_size = dims[axis];
    for (std::size_t i = axis + 1; i < dims.size(); ++i)  info.inner_size *= dims[i];
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i != axis) info.out_dims.push_back(dims[i]);
    }
    if (info.out_dims.empty()) info.out_dims.push_back(1);
    return info;
}

}
