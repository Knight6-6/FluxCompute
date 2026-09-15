#pragma once
#include <cstddef>

namespace flux::backend::cpu {

// 矩阵乘：C(M,N) = A(M,K) * B(K,N)，行主序连续存储。
//
// 这是 backend 里**第一个套不进 outer/axis/inner 三段分解的内核**。
// 三段分解适用于"沿某一维归约"，而 matmul 的输出同时依赖 A 的一整行与
// B 的一整列，是二维的访存模式，两者的循环结构不是一回事。
// 因此它自成一套：调用方直接给 m/k/n，不再传 outer/axis/inner。
//
// 循环序用 i-k-j 而不是教科书的 i-j-k：
//
//     i-j-k:  for i,j { for p: c[i][j] += a[i][p]*b[p][j] }   ← b 按列跳着走
//     i-k-j:  for i,p { for j: c[i][j] += a[i][p]*b[p][j] }   ← 三者都连续
//
// i-j-k 的内层是 `b[p*n + j]` 跨行访问（步长 n），缓存命中极差；i-k-j 把
// 标量 a[i][p] 提出来，内层对 b 的行与 c 的行都是连续访问。这不是什么
// 高级优化，只是"别选那个病态的循环序"——实测差距在数量级上。
//
// 代价是 c 需要先清零（原式是累加而非赋值）。
//
// 分块 / SIMD / 多线程都**没做**：按 README 原则 #4，那类优化要有基准支撑。
// 先有正确版本和基准数字，再谈怎么快。
template <typename T>
void matmul_kernel(const T* a, const T* b, T* c,
                   std::size_t m, std::size_t k, std::size_t n) {

    for (std::size_t i = 0; i < m; ++i) {
        T* crow = c + i * n;
        for (std::size_t j = 0; j < n; ++j) crow[j] = T{};
    }

    for (std::size_t i = 0; i < m; ++i) {
        const T* arow = a + i * k;
        T* crow = c + i * n;

        for (std::size_t p = 0; p < k; ++p) {
            const T aip = arow[p];
            const T* brow = b + p * n;

            #pragma omp simd
            for (std::size_t j = 0; j < n; ++j) {
                crow[j] += aip * brow[j];
            }
        }
    }
}

}
