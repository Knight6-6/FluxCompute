#pragma once
#include <cstddef>
#include <limits>
#include <algorithm>

namespace flux::backend::cpu {

template <typename T>
void shift_axis(const T* input, T* output, 
                std::size_t outer_size, 
                std::size_t axis_size, 
                std::size_t inner_size, 
                int offset, T fill_value) {
    
    for (std::size_t o = 0; o < outer_size; ++o) {
        for (std::size_t a = 0; a < axis_size; ++a) {

            int src_a = static_cast<int>(a) - offset;

            T* out_ptr = output + (o * axis_size + a) * inner_size;

            if (src_a >= 0 && src_a < static_cast<int>(axis_size)) {

                const T* in_ptr = input + (o * axis_size + src_a) * inner_size;
                std::copy_n(in_ptr, inner_size, out_ptr);
            } else {
                std::fill_n(out_ptr, inner_size, fill_value);
            }
        }
    }
}

}