#pragma once
#include <flux/tensor/tensor.hpp>
#include <flux/backend/cpu/shift.hpp>
#include <flux/operator/axis_util.hpp>
#include <cmath>
#include <limits>

namespace flux::ops {

template <typename T>
class Shift {
public:
    tensor::Tensor<T> operator()(const tensor::Tensor<T>& input, 
                                 int offset, 
                                 std::size_t axis = 0, 
                                 T fill_value = std::numeric_limits<T>::quiet_NaN()) const {
        
        const auto info = detail::compute_axis_info(input.shape(), axis);

        tensor::Tensor<T> result(input.shape());

        backend::cpu::shift_axis(
            input.data(), result.data(),
            info.outer_size, info.axis_size, info.inner_size,
            offset, fill_value
        );

        return result;
    }
};

template <typename T>
tensor::Tensor<T> shift(const tensor::Tensor<T>& input, 
                        int offset, 
                                std::size_t axis = 0, 
                        T fill_value = std::numeric_limits<T>::quiet_NaN()) {
    return Shift<T>{}(input, offset, axis, fill_value);
}

}