#pragma once
#include <cstddef>
#include <limits>
#include <cmath>

namespace flux::backend::cpu {

template <typename T>
void rolling_mean_axis(const T* input, T* output,
                        std::size_t outer_size,
                        std::size_t axis_size,
                        std::size_t inner_size,
                        std::size_t window,
                        std::size_t min_periods,
                        T fill_value) {
    
    for (std::size_t o = 0; o < outer_size; ++o) {
        for (std::size_t i = 0; i < inner_size; ++i) {

            const std::size_t stride = inner_size;
            const std::size_t base_offset = o * axis_size * inner_size + i;

            T current_sum = T{0};
            std::size_t valid_count = 0;

            for (std::size_t a = 0; a < axis_size; ++a) {
                const T new_val = input[base_offset + a * stride];
                
                if (!std::isnan(new_val)) {
                    current_sum += new_val;
                    valid_count++;
                }

                if (a >= window) {
                    const T old_val = input[base_offset + (a - window) * stride];
                    if (!std::isnan(old_val)) {
                        current_sum -= old_val;
                        valid_count--;
                    }
                }

                if (valid_count >= min_periods && valid_count > 0) {
                    output[base_offset + a * stride] = current_sum / static_cast<T>(valid_count);
                } else {
                    output[base_offset + a * stride] = fill_value;
                }
            }
        }
    }
}

}