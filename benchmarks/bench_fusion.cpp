// 算子融合值多少？
//
// 逐元素算子实测是**内存受限**的（多核并行只拿到 0.92x，见 bench_parallel），
// 所以真正的杠杆不是提高算力，而是减少内存流量。融合把 N 趟遍历压成 1 趟，
// 收益应当随链长线性增长——这里把它量出来。
//
// 用两种规模，因为融合的收益完全来自省下的内存流量，装得进缓存的规模下
// 收益应当明显变小。

#include <flux/flux.hpp>
#include "bench_util.hpp"

#include <cstdio>

using namespace flux;

namespace {

tensor::Tensor<float> make_data(std::size_t n, float seed) {
    tensor::Tensor<float> t(tensor::Shape({n}));
    for (std::size_t i = 0; i < n; ++i) {
        t[i] = static_cast<float>(i % 97) * seed + 1.0f;
    }
    return t;
}

void run_case(const char* label, std::size_t n) {
    const auto a = make_data(n, 0.5f);
    const auto b = make_data(n, 0.25f);
    const auto c = make_data(n, 0.125f);
    const auto d = make_data(n, 0.0625f);

    std::printf("  -- %s（%zu KB / 张量）--\n", label, n * sizeof(float) / 1024);

    const double two_unfused = flux_bench::best_ms(30, [&] {
        auto r = ops::mul(ops::add(a, b), c);
        flux_bench::keep(r.data());
    });
    const double two_fused = flux_bench::best_ms(30, [&] {
        auto r = ops::map3(a, b, c, [](float x, float y, float z) { return (x + y) * z; });
        flux_bench::keep(r.data());
    });

    const double three_unfused = flux_bench::best_ms(30, [&] {
        auto r = ops::sub(ops::mul(ops::add(a, b), c), d);
        flux_bench::keep(r.data());
    });
    const double three_fused = flux_bench::best_ms(30, [&] {
        auto r = ops::map4(a, b, c, d, [](float x, float y, float z, float w) {
            return (x + y) * z - w;
        });
        flux_bench::keep(r.data());
    });

    std::printf("     (a+b)*c     分离 %7.3f ms   融合 %7.3f ms   加速比 %5.2fx\n",
                two_unfused, two_fused, two_unfused / two_fused);
    std::printf("     (a+b)*c-d   分离 %7.3f ms   融合 %7.3f ms   加速比 %5.2fx\n\n",
                three_unfused, three_fused, three_unfused / three_fused);
}

} // namespace

int main() {
    std::printf("融合把 N 趟遍历压成 1 趟，收益来自省下的内存流量\n\n");

    run_case("超出 L3", 1u << 20);    // 4 MB
    run_case("装得进 L2", 1u << 15);  // 128 KB

    std::printf("参考：单算子基线\n");
    {
        const auto a = make_data(1u << 20, 0.5f);
        const auto b = make_data(1u << 20, 0.25f);
        flux_bench::bench("ops::add 1M（单个算子，一趟）", 30, [&] {
            auto r = ops::add(a, b);
            flux_bench::keep(r.data());
        });
    }
    return flux_bench::report("fusion");
}
