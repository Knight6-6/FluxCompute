// 矩阵乘的基线性能。
//
// 内核目前是朴素三重循环（只选了非病态的循环序 i-k-j），分块 / SIMD /
// 多线程都没做。这个基准的作用是**给出改造前的数字**——按 README 原则 #4，
// 值不值得优化、优化多少，都要拿它当参照。
//
// 用 GFLOPS 而不是毫秒：matmul 的工作量随尺寸立方增长，跨尺寸比毫秒没有
// 意义，而 GFLOPS 是可以和硬件峰值、和已知实现横向比的。
//
// 参考量级：现代 x86 单核朴素循环大致在 1~5 GFLOPS，调优过的 BLAS 单核
// 能到 50+ GFLOPS。差距不在这份代码要补的范围内——这里只是先把尺子立起来。

#include <flux/flux.hpp>
#include "bench_util.hpp"

#include <cstdio>

using namespace flux;

namespace {

tensor::Tensor<float> make_matrix(std::size_t r, std::size_t c) {
    tensor::Tensor<float> t(tensor::Shape({r, c}));
    for (std::size_t i = 0; i < t.numel(); ++i) {
        t[i] = static_cast<float>(i % 97) * 0.01f + 1.0f;
    }
    return t;
}

} // namespace

int main() {
    std::printf("朴素三重循环（i-k-j 循环序）。GFLOPS = 2*M*K*N / 时间\n\n");
    std::printf("  %-22s %10s %12s\n", "尺寸", "耗时", "GFLOPS");

    const std::size_t sizes[][3] = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
    };

    for (const auto& s : sizes) {
        const std::size_t m = s[0], k = s[1], n = s[2];
        const auto a = make_matrix(m, k);
        const auto b = make_matrix(k, n);

        const int reps = (m <= 256) ? 20 : 3;
        const double ms = flux_bench::best_ms(reps, [&] {
            auto c = ops::matmul(a, b);
            flux_bench::keep(c.data());
        });

        const double flops = 2.0 * static_cast<double>(m) * k * n;
        char name[64];
        std::snprintf(name, sizeof(name), "%zux%zux%zu", m, k, n);
        std::printf("  %-22s %8.3f ms %10.2f\n", name, ms, flops / ms / 1e6);
    }

    return 0;
}
