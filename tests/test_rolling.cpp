#include <flux/flux.hpp>
#include "test_util.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace flux;

namespace {

void test_rolling_mean_basic() {
    tensor::Tensor<float> t(tensor::Shape({5}), {1, 2, 3, 4, 5});
    auto r = ops::rolling_mean(t, 2);  // min_periods 默认 = window
    CHECK(std::isnan(r[0]));
    CHECK_NEAR(r[1], 1.5f, 1e-6f);
    CHECK_NEAR(r[2], 2.5f, 1e-6f);
    CHECK_NEAR(r[3], 3.5f, 1e-6f);
    CHECK_NEAR(r[4], 4.5f, 1e-6f);
}

void test_rolling_mean_min_periods() {
    // window=3，但只有 2 个有效数据即可输出
    tensor::Tensor<float> t(tensor::Shape({4}), {1, 2, 3, 4});
    auto r = ops::rolling_mean(t, 3, 2);
    CHECK(std::isnan(r[0]));          // 仅 1 个有效 < 2
    CHECK_NEAR(r[1], 1.5f, 1e-6f);    // {1,2}
    CHECK_NEAR(r[2], 2.0f, 1e-6f);    // {1,2,3}
    CHECK_NEAR(r[3], 3.0f, 1e-6f);    // {2,3,4}
}

void test_rolling_mean_nan_skip() {
    // NaN 不参与统计；窗口滑出后有效数下降
    tensor::Tensor<float> t(tensor::Shape({4}),
                            {1, std::numeric_limits<float>::quiet_NaN(), 3, 4});
    auto r = ops::rolling_mean(t, 2);
    CHECK(std::isnan(r[0]));          // 仅 {1}
    CHECK(std::isnan(r[1]));          // {1, NaN} -> 1 个有效
    CHECK(std::isnan(r[2]));          // {NaN, 3} -> 1 个有效 < min_periods
    CHECK_NEAR(r[3], 3.5f, 1e-6f);    // {3, 4}
}

void test_invalid_window() {
    tensor::Tensor<float> t(tensor::Shape({3}), {1, 2, 3});
    bool threw = false;
    try {
        (void)ops::rolling_mean(t, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_rolling_std_basic() {
    // 样本标准差（ddof=1）：所有窗口都是相邻两数，标准差都是 sqrt(0.5)
    tensor::Tensor<float> t(tensor::Shape({5}), {1, 2, 3, 4, 5});
    auto r = ops::rolling_std(t, 2);   // min_periods 默认 = window = 2
    CHECK(std::isnan(r[0]));           // 只有 1 个有效样本，n <= ddof
    for (std::size_t i = 1; i < 5; ++i) CHECK_NEAR(r[i], std::sqrt(0.5f), 1e-6f);
}

void test_rolling_std_ddof() {
    // ddof=1 是样本标准差，ddof=0 是总体标准差：{1,3} 分别为 sqrt(2) 与 1
    tensor::Tensor<float> t(tensor::Shape({2}), {1, 3});

    auto sample = ops::rolling_std(t, 2);
    CHECK(std::isnan(sample[0]));   // n=1 <= ddof=1，无定义
    CHECK_NEAR(sample[1], std::sqrt(2.f), 1e-6f);

    // 注意 min_periods 传 0 会被默认成 window（=2），那样第一个位置仍是 NaN。
    // 要观察 ddof=0 在单样本上的行为，必须显式给 min_periods=1。
    auto pop = ops::rolling_std(t, 2, /*min_periods=*/1, 0,
                                std::numeric_limits<float>::quiet_NaN(), /*ddof=*/0);
    CHECK_NEAR(pop[0], 0.f, 1e-6f);   // 单样本的总体标准差为 0
    CHECK_NEAR(pop[1], 1.f, 1e-6f);
}

void test_rolling_std_constant_series_is_zero() {
    tensor::Tensor<float> t(tensor::Shape({5}), {7, 7, 7, 7, 7});
    auto r = ops::rolling_std(t, 3);
    CHECK(std::isnan(r[0]));
    CHECK(std::isnan(r[1]));
    CHECK_NEAR(r[2], 0.f, 1e-6f);
    CHECK_NEAR(r[4], 0.f, 1e-6f);
}

void test_rolling_std_min_periods() {
    tensor::Tensor<float> t(tensor::Shape({4}), {1, 2, 3, 4});
    auto r = ops::rolling_std(t, 3, /*min_periods=*/2);
    CHECK(std::isnan(r[0]));
    CHECK_NEAR(r[1], std::sqrt(0.5f), 1e-6f);   // {1,2}
    CHECK_NEAR(r[2], 1.f, 1e-6f);               // {1,2,3}
    CHECK_NEAR(r[3], 1.f, 1e-6f);               // {2,3,4}
}

void test_rolling_std_nan_skip() {
    tensor::Tensor<float> t(tensor::Shape({4}),
                            {1, std::numeric_limits<float>::quiet_NaN(), 3, 4});
    auto r = ops::rolling_std(t, 2);
    CHECK(std::isnan(r[0]));                    // {1}
    CHECK(std::isnan(r[1]));                    // {1, NaN}
    CHECK(std::isnan(r[2]));                    // {NaN, 3}
    CHECK_NEAR(r[3], std::sqrt(0.5f), 1e-6f);   // {3, 4}
}

void test_rolling_std_axis() {
    // {3,2} [1,2, 3,4, 5,6] 沿 axis1，每行相邻两数 -> 标准差 sqrt(0.5)
    tensor::Tensor<float> t(tensor::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    auto r = ops::rolling_std(t, 2, 0, /*axis=*/1);
    CHECK(std::isnan(r[0]));
    CHECK_NEAR(r[1], std::sqrt(0.5f), 1e-6f);
    CHECK(std::isnan(r[2]));
    CHECK_NEAR(r[3], std::sqrt(0.5f), 1e-6f);
    CHECK(std::isnan(r[4]));
    CHECK_NEAR(r[5], std::sqrt(0.5f), 1e-6f);
}

void test_rolling_std_accuracy_on_realistic_magnitudes() {
    // 数值回归：累加器若退回 float，成交额这种量级会因 sum/sumsq 相减的
    // 灾难性抵消而严重失真（实测相对误差可达 1.0e+00，等于没算）。
    // 参照用稳定的两趟算法（先减均值）在 double 下算，作为真值。
    const std::size_t n = 64, w = 8;
    tensor::Tensor<float> t(tensor::Shape({n}));
    for (std::size_t i = 0; i < n; ++i) {
        t[i] = 1e6f + static_cast<float>(i % 7) * 100.f;
    }

    auto got = ops::rolling_std(t, w);

    for (std::size_t i = w - 1; i < n; ++i) {
        double sum = 0;
        for (std::size_t k = i + 1 - w; k <= i; ++k) sum += static_cast<double>(t[k]);
        const double mean = sum / static_cast<double>(w);

        double ss = 0;
        for (std::size_t k = i + 1 - w; k <= i; ++k) {
            const double d = static_cast<double>(t[k]) - mean;
            ss += d * d;
        }
        const float expected = static_cast<float>(std::sqrt(ss / static_cast<double>(w - 1)));

        CHECK_NEAR(got[i], expected, expected * 1e-4f);   // 相对 1e-4
    }
}

void test_rolling_std_invalid_window() {
    tensor::Tensor<float> t(tensor::Shape({3}), {1, 2, 3});
    bool threw = false;
    try {
        (void)ops::rolling_std(t, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_axis_out_of_range() {
    tensor::Tensor<float> t(tensor::Shape({2, 2}), {1, 2, 3, 4});
    bool threw = false;
    try {
        (void)ops::rolling_mean(t, 2, 2, 7);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    test_rolling_mean_basic();
    test_rolling_mean_min_periods();
    test_rolling_mean_nan_skip();
    test_rolling_std_basic();
    test_rolling_std_ddof();
    test_rolling_std_constant_series_is_zero();
    test_rolling_std_min_periods();
    test_rolling_std_nan_skip();
    test_rolling_std_axis();
    test_rolling_std_accuracy_on_realistic_magnitudes();
    test_rolling_std_invalid_window();
    test_invalid_window();
    test_axis_out_of_range();
    return flux_test::summary();
}
