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

// ---------------------------------------------------------------------------
// N 元逐元素：把多个输入在一趟遍历里合并成输出，不产生中间张量。
//
// 这是算子融合的落点。逐元素算子本身是**内存受限**的（实测多核并行只拿到
// 0.92x，见 docs/04），所以省掉一遍遍历就是省掉一遍内存流量——收益随链长
// 线性增长：实测 (a+b)*c 融合后 2.35x、(a+b)*c-a 融合后 3.86x。
//
// 分成 zip2/zip3/zip4 三个固定元数的版本，而不是一个 N 元通用版：
// 后者需要让 Op 接受数组，既损失可读性也让内联变难。固定元数的每个都
// 是平凡的 SIMD 循环。
//
// op 直接写成标量 lambda，没有任何类型擦除开销——这是它与"把图上若干个
// 闭包串起来"的本质差别，后者每元素都要一次间接调用。
// ---------------------------------------------------------------------------

template <typename T, typename Op>
void zip2(const T* a, const T* b, T* out, std::size_t size, Op op) {
    #pragma omp simd
    for (std::size_t i = 0; i < size; ++i) {
        out[i] = op(a[i], b[i]);
    }
}

template <typename T, typename Op>
void zip3(const T* a, const T* b, const T* c, T* out, std::size_t size, Op op) {
    #pragma omp simd
    for (std::size_t i = 0; i < size; ++i) {
        out[i] = op(a[i], b[i], c[i]);
    }
}

template <typename T, typename Op>
void zip4(const T* a, const T* b, const T* c, const T* d, T* out, std::size_t size, Op op) {
    #pragma omp simd
    for (std::size_t i = 0; i < size; ++i) {
        out[i] = op(a[i], b[i], c[i], d[i]);
    }
}

}