#pragma once

#include <flux/tensor/shape.hpp>
#include <flux/memory/aligned_allocator.hpp>
#include <flux/memory/memory_pool.hpp>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cstddef>
#include <memory>

namespace flux::tensor {

template <typename T>
class Tensor {
public:
    using ContainerType = std::vector<T, flux::memory::PooledAllocator<T>>;

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

// 计算图中边上传递的共享句柄。一次计算只物化一次，下游共享同一份数据。
//
// const 承载的是不变量，不是防御性写法：Executor 的零拷贝安全性完全建立在
// "张量一旦发布即不可变"之上——消费者拿到的若可变，扇出就成了别名 bug。
// 因此只提供这一个别名，不要另加 shared_ptr<Tensor<T>>（可变）版本。
template <typename T>
using TensorPtr = std::shared_ptr<const Tensor<T>>;

}