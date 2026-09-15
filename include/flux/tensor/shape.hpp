#pragma once

#include <vector>
#include <cstddef>

namespace flux::tensor {

class Shape {
private:
    std::vector<std::size_t> dimensions_;
    std::size_t numel_;
    std::vector<std::size_t> strides_;

public:
    Shape(const std::vector<std::size_t>& dimensions) : dimensions_(dimensions), numel_(1) {
        for (const auto& dim : dimensions_) {
            numel_ *= dim;
        }
        
        auto stride = numel_;
        for (std::size_t i = 0; i < dimensions_.size(); ++i) {
            // 任一维为 0 时 numel_ 已经是 0，该维的 stride 语义上无定义
            // （根本没有元素可寻址）。直接做 0/0 是整数除零，会 SIGFPE 干掉
            // 整个进程——而零长度维度在真实数据里很常见（空交易日、
            // 没有标的通过筛选）。显式置 0，让空张量能正常构造与传递。
            stride = (dimensions_[i] == 0) ? 0 : (stride / dimensions_[i]);
            strides_.push_back(stride);
        }
    }

    std::size_t numel() const {
        return numel_;
    }

    const std::vector<std::size_t>& dimensions() const {
        return dimensions_;
    }

    std::size_t stride(std::size_t index) const {
        return strides_[index];
    }

    bool operator==(const Shape& other) const {
        return dimensions_ == other.dimensions_;
    }

    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }
};

}