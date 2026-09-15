#include <flux/flux.hpp>
#include "test_util.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

using namespace flux;

namespace {

auto max_op = [](float a, float b) { return a > b ? a : b; };

void test_reduce_all_plus_matches_sum() {
    // 干净数据（无 NaN）上，通用归约 + std::plus 应当与专用 sum 一致
    tensor::Tensor<float> t(tensor::Shape({5}), {1, 2, 3, 4, 5});
    CHECK_NEAR(ops::reduce(t, std::plus<float>{}, 0.f), ops::sum(t), 1e-6f);
}

void test_reduce_all_max() {
    tensor::Tensor<float> t(tensor::Shape({5}), {3, 7, 2, 9, 5});
    CHECK_EQ(ops::reduce(t, max_op, std::numeric_limits<float>::lowest()), 9.f);
}

void test_reduce_axis0_2d() {
    // shape {3,2} 行主序 [1,2, 3,4, 5,6]，按 axis0 折叠 -> [9, 12]
    tensor::Tensor<float> t(tensor::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    auto r = ops::reduce(t, 0, std::plus<float>{}, 0.f);
    CHECK(r.shape() == tensor::Shape({2}));
    CHECK_EQ(r[0], 9.f);
    CHECK_EQ(r[1], 12.f);
}

void test_reduce_axis1_2d() {
    tensor::Tensor<float> t(tensor::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    auto r = ops::reduce(t, 1, std::plus<float>{}, 0.f);
    CHECK(r.shape() == tensor::Shape({3}));
    CHECK_EQ(r[0], 3.f);
    CHECK_EQ(r[1], 7.f);
    CHECK_EQ(r[2], 11.f);
}

void test_reduce_axis0_3d() {
    tensor::Tensor<float> t(tensor::Shape({2, 2, 3}), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto r = ops::reduce(t, 0, std::plus<float>{}, 0.f);
    CHECK(r.shape() == tensor::Shape({2, 3}));
    // 沿 axis0 把两个 batch 的对应位置相加：第 i 个输出是 (i+1) + (i+7)。
    // 注意别照抄 test_max.cpp 的 7+i —— 那是求最大值的公式，不是求和。
    for (std::size_t i = 0; i < 6; ++i) {
        CHECK_EQ(r[i], static_cast<float>(i + 1) + static_cast<float>(i + 7));
    }
}

void test_reduce_int() {
    tensor::Tensor<int> t(tensor::Shape({3}), {1, 2, 3});
    CHECK_EQ(ops::reduce(t, std::plus<int>{}, 0), 6);
}

void test_reduce_identity_is_not_optional() {
    // 这条测试解释了单位元为什么必须是显式参数而非 T{} 兜底：
    // 全负切片上求最大值，用 0 当单位元会得到错误的 0。
    // 同一个陷阱在 extremum 处踩过一次（见 test_max_int_negatives）。
    tensor::Tensor<int> t(tensor::Shape({4}), {-5, -1, -3, -2});
    auto imax = [](int a, int b) { return a > b ? a : b; };

    CHECK_EQ(ops::reduce(t, imax, std::numeric_limits<int>::lowest()), -1);
    CHECK_EQ(ops::reduce(t, imax, 0), 0);   // 反面教材：0 兜底 -> 错
}

void test_reduce_does_not_skip_nan() {
    // 通用归约不跳过 NaN：它拿到的是任意算子，无从知道 NaN 该怎么处理。
    // 对比同一份数据上 sum 的 skipna 行为——这两个语义**故意不同**，
    // 与 pandas 一致（sum/mean 默认 skipna，agg 传自定义函数则不生效）。
    tensor::Tensor<float> t(tensor::Shape({4}),
                            {1, std::numeric_limits<float>::quiet_NaN(), 3, 4});

    CHECK_NEAR(ops::sum(t), 8.f, 1e-6f);
    CHECK(std::isnan(ops::reduce(t, std::plus<float>{}, 0.f)));
}

void test_reduce_is_a_left_fold() {
    // 折叠顺序固定为从左到右，对非交换算子结果确定：
    // ((0 - 1) - 2) - 3 - 4 = -10
    tensor::Tensor<float> t(tensor::Shape({4}), {1, 2, 3, 4});
    auto sub = [](float a, float b) { return a - b; };
    CHECK_NEAR(ops::reduce(t, sub, 0.f), -10.f, 1e-6f);
}

void test_reduce_accepts_capturing_functor() {
    // 算子可以是带捕获的闭包，不必是无捕获 lambda
    tensor::Tensor<float> t(tensor::Shape({5}), {1, 2, 3, 4, 5});
    const float scale = 2.f;
    auto add_scaled = [scale](float a, float b) { return a + b * scale; };
    CHECK_NEAR(ops::reduce(t, add_scaled, 0.f), 30.f, 1e-6f);
}

void test_reduce_axis_out_of_range() {
    tensor::Tensor<float> t(tensor::Shape({2, 2}), {1, 2, 3, 4});
    bool threw = false;
    try {
        (void)ops::reduce(t, 5, std::plus<float>{}, 0.f);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    test_reduce_all_plus_matches_sum();
    test_reduce_all_max();
    test_reduce_axis0_2d();
    test_reduce_axis1_2d();
    test_reduce_axis0_3d();
    test_reduce_int();
    test_reduce_identity_is_not_optional();
    test_reduce_does_not_skip_nan();
    test_reduce_is_a_left_fold();
    test_reduce_accepts_capturing_functor();
    test_reduce_axis_out_of_range();
    return flux_test::summary();
}
