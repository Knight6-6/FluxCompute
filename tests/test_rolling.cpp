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
    test_invalid_window();
    test_axis_out_of_range();
    return flux_test::summary();
}
