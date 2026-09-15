#pragma once

#include <flux/operator/binary_elementwise.hpp>

namespace flux::ops {

struct MulOperation {
    template <typename T>
    T operator()(T a, T b) const {
        return a * b;
    }
};

template <typename T>
class Mul {
public:
    tensor::Tensor<T> operator()(const tensor::Tensor<T>& a, const tensor::Tensor<T>& b) const {
        BinaryElementwise<T, MulOperation> operation;
        return operation(a, b);
    }
};

template <typename T>
tensor::Tensor<T> mul(const tensor::Tensor<T>& a, const tensor::Tensor<T>& b) {
    return Mul<T>{}(a, b);
}

}