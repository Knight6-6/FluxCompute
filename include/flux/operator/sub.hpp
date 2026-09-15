#pragma once
#include <flux/operator/binary_elementwise.hpp>

namespace flux::ops {

struct SubOperation {
    template <typename T>
    T operator()(T a, T b) const { return a - b; }
};

template <typename T>
tensor::Tensor<T> sub(const tensor::Tensor<T>& a, const tensor::Tensor<T>& b) {
    return BinaryElementwise<T, SubOperation>{}(a, b);
}

}