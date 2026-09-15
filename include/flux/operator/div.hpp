#pragma once
#include <flux/operator/binary_elementwise.hpp>

namespace flux::ops {

struct DivOperation {
    template <typename T>
    T operator()(T a, T b) const { return a / b; }
};

template <typename T>
tensor::Tensor<T> div(const tensor::Tensor<T>& a, const tensor::Tensor<T>& b) {
    return BinaryElementwise<T, DivOperation>{}(a, b);
}

}