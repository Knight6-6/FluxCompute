#pragma once
#include <cstddef>

namespace flux::backend::cpu {

template <typename T, typename Op>
void binary_elementwise(const T* a, const T* b, T* out, std::size_t size) {
    #pragma omp simd
    for (std::size_t i = 0; i < size; ++i) {
        out[i] = Op{}(a[i], b[i]);
    }
}

}