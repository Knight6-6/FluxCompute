#pragma once

#include <flux/tensor/shape.hpp>
#include <flux/memory/aligned_allocator.hpp>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cstddef>

namespace flux::tensor {

template <typename T>
class Tensor {
public:
    using ContainerType = std::vector<T, flux::memory::AlignedAllocator<T, 64>>;

private:
    Shape shape_;
    ContainerType data_;

public:
    explicit Tensor(const Shape& shape) : shape_(shape), data_(shape.numel()) {}

    Tensor(const Shape& shape, ContainerType data) : shape_(shape), data_(std::move(data)) {
        if (data_.size() != shape_.numel()) {
            throw std::invalid_argument("Data size does not match tensor shape");
        }
    }

    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    const Shape& shape() const { return shape_; }
    std::size_t numel() const { return data_.size(); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    T& operator[](std::size_t idx) { return data_[idx]; }
    const T& operator[](std::size_t idx) const { return data_[idx]; }
};

}