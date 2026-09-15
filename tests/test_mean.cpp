#include <flux/flux.hpp>
#include "test_util.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace flux;

namespace {

void test_mean_all() {
    tensor::Tensor<float> t(tensor::Shape({5}), {1, 2, 3, 4, 5});
    CHECK_NEAR(ops::mean(t), 3.f, 1e-6f);
}

void test_mean_axis0_2d() {
    // shape {3,2}: [1,2, 3,4, 5,6]，按 axis0 求均值 -> [3,4]
    tensor::Tensor<float> t(tensor::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    auto r = ops::mean(t, 0);
    CHECK(r.shape() == tensor::Shape({2}));
    CHECK_NEAR(r[0], 3.f, 1e-6f);
    CHECK_NEAR(r[1], 4.f, 1e-6f);
}

void test_mean_axis1_2d() {
    // 按 axis1 求均值 -> [1.5, 3.5, 5.5]
    tensor::Tensor<float> t(tensor::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    auto r = ops::mean(t, 1);
    CHECK(r.shape() == tensor::Shape({3}));
    CHECK_NEAR(r[0], 1.5f, 1e-6f);
    CHECK_NEAR(r[1], 3.5f, 1e-6f);
    CHECK_NEAR(r[2], 5.5f, 1e-6f);
}

void test_mean_axis0_3d() {
    // shape {2,2,3}: [1..12]，按 axis0 求均值 -> [4,5,6, 7,8,9]
    tensor::Tensor<float> t(tensor::Shape({2, 2, 3}), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto r = ops::mean(t, 0);
    CHECK(r.shape() == tensor::Shape({2, 3}));
    for (std::size_t i = 0; i < 6; ++i) CHECK_NEAR(r[i], 4.f + i, 1e-6f);
}

void test_mean_axis1_3d() {
    tensor::Tensor<float> t(tensor::Shape({2, 2, 3}), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto r = ops::mean(t, 1);
    CHECK(r.shape() == tensor::Shape({2, 3}));
    CHECK_NEAR(r[0], 2.5f, 1e-6f);
    CHECK_NEAR(r[1], 3.5f, 1e-6f);
    CHECK_NEAR(r[2], 4.5f, 1e-6f);
    CHECK_NEAR(r[3], 8.5f, 1e-6f);
    CHECK_NEAR(r[4], 9.5f, 1e-6f);
    CHECK_NEAR(r[5], 10.5f, 1e-6f);
}

void test_mean_nan_skip() {
    // NaN 不参与统计
    tensor::Tensor<float> t(tensor::Shape({5}), {1, std::numeric_limits<float>::quiet_NaN(),
                                                 2, 3, std::numeric_limits<float>::quiet_NaN()});
    CHECK_NEAR(ops::mean(t), 2.f, 1e-6f);
}

void test_mean_all_nan() {
    tensor::Tensor<float> t(tensor::Shape({2}), {std::numeric_limits<float>::quiet_NaN(),
                                                 std::numeric_limits<float>::quiet_NaN()});
    CHECK(std::isnan(ops::mean(t)));
}

void test_mean_all_nan_axis() {
    // {2,2}: 第 0 行全 NaN -> NaN，第 1 行 {1,2} -> 1.5
    tensor::Tensor<float> t(tensor::Shape({2, 2}),
                            {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             1, 2});
    auto r = ops::mean(t, 1);
    CHECK(std::isnan(r[0]));
    CHECK_NEAR(r[1], 1.5f, 1e-6f);
}

void test_mean_int_truncation() {
    // 整数 T 是整除截断：{1,2,3,4} -> 10/4 = 2
    tensor::Tensor<int> t(tensor::Shape({4}), {1, 2, 3, 4});
    CHECK_EQ(ops::mean(t), 2);
}

void test_mean_axis_int() {
    tensor::Tensor<int> t(tensor::Shape({2, 2}), {1, 2, 3, 4});
    auto r = ops::mean(t, 1);
    CHECK_EQ(r[0], 1);  // {1,2} -> 1
    CHECK_EQ(r[1], 3);  // {3,4} -> 3
}

void test_mean_axis_out_of_range() {
    tensor::Tensor<float> t(tensor::Shape({2, 2}), {1, 2, 3, 4});
    bool threw = false;
    try {
        (void)ops::mean(t, 5);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    test_mean_all();
    test_mean_axis0_2d();
    test_mean_axis1_2d();
    test_mean_axis0_3d();
    test_mean_axis1_3d();
    test_mean_nan_skip();
    test_mean_all_nan();
    test_mean_all_nan_axis();
    test_mean_int_truncation();
    test_mean_axis_int();
    test_mean_axis_out_of_range();
    return flux_test::summary();
}
