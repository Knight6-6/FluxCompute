#include <flux/flux.hpp>
#include "test_util.hpp"

#include <cmath>
#include <stdexcept>

using namespace flux;

namespace {

void test_shift_lag_1d() {
    // shift(1)：数据向下平移（Lag 1），头部补 NaN
    tensor::Tensor<float> t(tensor::Shape({5}), {10, 11, 12, 13, 14});
    auto r = ops::shift(t, 1);
    CHECK(std::isnan(r[0]));
    CHECK_EQ(r[1], 10.f);
    CHECK_EQ(r[2], 11.f);
    CHECK_EQ(r[3], 12.f);
    CHECK_EQ(r[4], 13.f);
}

void test_shift_lead_1d() {
    // shift(-1)：数据向上平移（Lead 1），尾部补 NaN
    tensor::Tensor<float> t(tensor::Shape({4}), {1, 2, 3, 4});
    auto r = ops::shift(t, -1);
    CHECK_EQ(r[0], 2.f);
    CHECK_EQ(r[1], 3.f);
    CHECK_EQ(r[2], 4.f);
    CHECK(std::isnan(r[3]));
}

void test_shift_custom_fill() {
    tensor::Tensor<float> t(tensor::Shape({3}), {1, 2, 3});
    auto r = ops::shift(t, 1, 0, 0.f);
    CHECK_EQ(r[0], 0.f);
    CHECK_EQ(r[1], 1.f);
    CHECK_EQ(r[2], 2.f);
}

void test_shift_axis1_2d() {
    // shape {3,2}：每行沿 axis1 右移 1 -> [nan,1, nan,3, nan,5]
    tensor::Tensor<float> t(tensor::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    auto r = ops::shift(t, 1, 1);
    CHECK(std::isnan(r[0]));
    CHECK_EQ(r[1], 1.f);
    CHECK(std::isnan(r[2]));
    CHECK_EQ(r[3], 3.f);
    CHECK(std::isnan(r[4]));
    CHECK_EQ(r[5], 5.f);
}

void test_shift_axis_out_of_range() {
    tensor::Tensor<float> t(tensor::Shape({2, 2}), {1, 2, 3, 4});
    bool threw = false;
    try {
        (void)ops::shift(t, 1, 7);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    test_shift_lag_1d();
    test_shift_lead_1d();
    test_shift_custom_fill();
    test_shift_axis1_2d();
    test_shift_axis_out_of_range();
    return flux_test::summary();
}
