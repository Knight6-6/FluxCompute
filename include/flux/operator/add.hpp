#pragma once

#include <flux/operator/binary_elementwise.hpp>

namespace flux::ops {

struct AddOperation {
    template <typename T>
    T operator()(T a, T b) const {
        return a + b;
    }
};

template <typename T>
class Add {
public:
    tensor::Tensor<T> operator()(const tensor::Tensor<T>& a, const tensor::Tensor<T>& b) const {
        BinaryElementwise<T, AddOperation> operation;
        return operation(a, b);
    }
};

template <typename T>
tensor::Tensor<T> add(const tensor::Tensor<T>& a, const tensor::Tensor<T>& b) {
    return Add<T>{}(a, b);
}

}