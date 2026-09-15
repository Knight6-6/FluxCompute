#include <flux/flux.hpp>
#include "test_util.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

using namespace flux;

namespace {

auto max_op = [](float a, float b) { return a > b ? a : b; };

void test_scan_cumsum_1d() {
    tensor::Tensor<float> t(tensor::Shape({4}), {1, 2, 3, 4});
    auto r = ops::scan(t, std::plus<float>{}, 0.f);
    CHECK(r.shape() == tensor::Shape({4}));
    CHECK_EQ(r[0], 1.f);
    CHECK_EQ(r[1], 3.f);
    CHECK_EQ(r[2], 6.f);
    CHECK_EQ(r[3], 10.f);
}

void test_scan_cummax_1d() {
    tensor::Tensor<float> t(tensor::Shape({5}), {3, 1, 4, 1, 5});
    auto r = ops::scan(t, max_op, std::numeric_limits<float>::lowest());
    CHECK_EQ(r[0], 3.f);
    CHECK_EQ(r[1], 3.f);
    CHECK_EQ(r[2], 4.f);
    CHECK_EQ(r[3], 4.f);
    CHECK_EQ(r[4], 5.f);
}

void test_scan_output_shape_matches_input() {
    // 与 reduce 的差别之一：scan 保留被扫描的轴
    tensor::Tensor<float> t(tensor::Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    auto r = ops::scan(t, std::plus<float>{}, 0.f, 0);
    CHECK(r.shape() == t.shape());
    CHECK(r.shape() == tensor::Shape({2, 3}));
}

void test_scan_axis0_2d() {
    // {3,2} 行主序 [1,2, 3,4, 5,6]，沿 axis0 前缀和 -> [1,2, 4,6, 9,12]
    tensor::Tensor<float> t(tensor::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    auto r = ops::scan(t, std::plus<float>{}, 0.f, 0);
    const float expected[] = {1, 2, 4, 6, 9, 12};
    for (std::size_t i = 0; i < 6; ++i) CHECK_EQ(r[i], expected[i]);
}

void test_scan_axis1_2d() {
    // 沿 axis1 前缀和 -> [1,3, 3,7, 5,11]
    tensor::Tensor<float> t(tensor::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    auto r = ops::scan(t, std::plus<float>{}, 0.f, 1);
    const float expected[] = {1, 3, 3, 7, 5, 11};
    for (std::size_t i = 0; i < 6; ++i) CHECK_EQ(r[i], expected[i]);
}

void test_scan_int_cumsum() {
    tensor::Tensor<int> t(tensor::Shape({4}), {1, 2, 3, 4});
    auto r = ops::scan(t, std::plus<int>{}, 0);
    CHECK_EQ(r[0], 1);
    CHECK_EQ(r[3], 10);
}

void test_scan_does_not_skip_nan() {
    // 与 reduce 同一约定：不跳过 NaN。plus 一旦吃到 NaN，之后全是 NaN。
    //
    // 这条**故意与 pandas cumsum 不同**——pandas 在 NaN 处输出 NaN 但继续
    // 累加（[1, NaN, 3] -> [1, NaN, 4]）。要那种语义需要在算子层显式处理，
    // 不该把假设埋进通用内核。这里把差异钉成断言，免得被当成 bug。
    tensor::Tensor<float> t(tensor::Shape({3}),
                            {1, std::numeric_limits<float>::quiet_NaN(), 3});
    auto r = ops::scan(t, std::plus<float>{}, 0.f);
    CHECK_EQ(r[0], 1.f);
    CHECK(std::isnan(r[1]));
    CHECK(std::isnan(r[2]));
}

void test_scan_identity_is_not_optional() {
    // 同 reduce：单位元给错，第一个元素就错。全负切片上求累计最大，
    // 用 0 当单位元会得到 [0, 0, 0]。
    tensor::Tensor<int> t(tensor::Shape({3}), {-5, -1, -3});
    auto imax = [](int a, int b) { return a > b ? a : b; };

    auto ok = ops::scan(t, imax, std::numeric_limits<int>::lowest());
    CHECK_EQ(ok[0], -5);
    CHECK_EQ(ok[1], -1);
    CHECK_EQ(ok[2], -1);

    auto bad = ops::scan(t, imax, 0);   // 反面教材
    CHECK_EQ(bad[0], 0);
    CHECK_EQ(bad[1], 0);
    CHECK_EQ(bad[2], 0);
}

void test_scan_is_a_left_fold() {
    // 折叠顺序固定从左到右：((0-1)-2)-3-4 的每个前缀
    tensor::Tensor<float> t(tensor::Shape({4}), {1, 2, 3, 4});
    auto sub = [](float a, float b) { return a - b; };
    auto r = ops::scan(t, sub, 0.f);
    CHECK_NEAR(r[0], -1.f, 1e-6f);
    CHECK_NEAR(r[1], -3.f, 1e-6f);
    CHECK_NEAR(r[2], -6.f, 1e-6f);
    CHECK_NEAR(r[3], -10.f, 1e-6f);
}

void test_scan_accepts_capturing_functor() {
    tensor::Tensor<float> t(tensor::Shape({3}), {1, 2, 3});
    const float scale = 10.f;
    auto add_scaled = [scale](float a, float b) { return a + b * scale; };
    auto r = ops::scan(t, add_scaled, 0.f);
    CHECK_NEAR(r[0], 10.f, 1e-6f);
    CHECK_NEAR(r[1], 30.f, 1e-6f);
    CHECK_NEAR(r[2], 60.f, 1e-6f);
}

void test_scan_axis_out_of_range() {
    tensor::Tensor<float> t(tensor::Shape({2, 2}), {1, 2, 3, 4});
    bool threw = false;
    try {
        (void)ops::scan(t, std::plus<float>{}, 0.f, 5);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    test_scan_cumsum_1d();
    test_scan_cummax_1d();
    test_scan_output_shape_matches_input();
    test_scan_axis0_2d();
    test_scan_axis1_2d();
    test_scan_int_cumsum();
    test_scan_does_not_skip_nan();
    test_scan_identity_is_not_optional();
    test_scan_is_a_left_fold();
    test_scan_accepts_capturing_functor();
    test_scan_axis_out_of_range();
    return flux_test::summary();
}
