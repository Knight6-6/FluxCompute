// `#pragma omp simd` 到底有没有用？
//
// 这是项目从建立起就在做的假设——backend/cpu/ 里几处内核都挂了 simd 提示，
// 但从未验证过收益。缺 -fopenmp 时这些 pragma 会被编译器忽略，所以把同一个
// 源文件编成两个目标（链 flux 与 flux_no_omp），两者的差就是提示的净效果。
//
// 用两种规模，因为 SIMD 的收益强烈依赖数据是否装得下缓存：
//   - 小规模（256KB，装得进 L2）：计算受限，SIMD 应当有明显收益；
//   - 大规模（16MB，超出 L3）：内存带宽受限，SIMD 可能几乎没有收益
//     ——算得再快也得等内存，这个结论同样重要。

#include <flux/flux.hpp>
#include "bench_util.hpp"

#include <cstdio>
#include <vector>

using namespace flux;

namespace {

constexpr std::size_t kSmall = 1u << 16;   // 64K float = 256 KB，装得进 L2
constexpr std::size_t kLarge = 1u << 22;   // 4M float = 16 MB，超出 L3

tensor::Tensor<float> make_data(std::size_t n) {
    tensor::Tensor<float> t(tensor::Shape({n}));
    for (std::size_t i = 0; i < n; ++i) t[i] = static_cast<float>(i % 97) * 0.5f + 1.0f;
    return t;
}

} // namespace

int main() {
#ifdef _OPENMP
    std::printf("本目标 **开启** 了 OpenMP：`#pragma omp simd` 生效\n");
#else
    std::printf("本目标 **未开启** OpenMP：`#pragma omp simd` 被编译器忽略\n");
#endif

    const auto small = make_data(kSmall);
    const auto large = make_data(kLarge);

    // --- 算子层：包含输出分配，是使用者实际付的成本 ---
    flux_bench::bench("算子 ops::add   256KB", 100, [&] {
        auto r = ops::add(small, small);
        flux_bench::keep(r.data());
    });
    flux_bench::bench("算子 ops::add   16MB ", 30, [&] {
        auto r = ops::add(large, large);
        flux_bench::keep(r.data());
    });

    // --- 内核层：绕开分配，隔离出纯计算，这才是 simd 提示的作用面 ---
    {
        tensor::Tensor<float> out(tensor::Shape({kSmall}));
        flux_bench::bench("内核 binary_elementwise 256KB", 200, [&] {
            backend::cpu::binary_elementwise<float, ops::AddOperation>(
                small.data(), small.data(), out.data(), kSmall);
            flux_bench::keep(out.data());
        });
    }
    {
        tensor::Tensor<float> out(tensor::Shape({kLarge}));
        flux_bench::bench("内核 binary_elementwise 16MB ", 30, [&] {
            backend::cpu::binary_elementwise<float, ops::AddOperation>(
                large.data(), large.data(), out.data(), kLarge);
            flux_bench::keep(out.data());
        });
    }

    // --- 归约：sum_all 的 reduction 子句 ---
    flux_bench::bench("内核 sum_all 256KB", 200, [&] {
        volatile float s = backend::cpu::sum_all(small.data(), kSmall);
        (void)s;
    });
    flux_bench::bench("内核 sum_all 16MB ", 30, [&] {
        volatile float s = backend::cpu::sum_all(large.data(), kLarge);
        (void)s;
    });

    return flux_bench::report("elementwise / reduction");
}
