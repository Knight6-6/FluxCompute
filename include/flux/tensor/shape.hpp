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
            stride /= dimensions_[i];
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