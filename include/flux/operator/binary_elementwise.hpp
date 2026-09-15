#pragma once
#include <flux/tensor/tensor.hpp>
#include <flux/backend/cpu/elementwise.hpp>
#include <stdexcept>

namespace flux::ops {

template <typename T, typename Operation>
class BinaryElementwise {
public:
    tensor::Tensor<T> operator()(const tensor::Tensor<T>& a, const tensor::Tensor<T>& b) const {
        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Shapes of tensors must match for binary operation");
        }
        tensor::Tensor<T> result(a.shape());
        
        backend::cpu::binary_elementwise<T, Operation>(
            a.data(), b.data(), result.data(), a.numel()
        );
        
        return result;
    }
};

}