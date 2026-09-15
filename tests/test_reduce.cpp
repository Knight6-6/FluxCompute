#include <flux/flux.hpp>
#include "test_util.hpp"

#include <limits>
#include <stdexcept>

using namespace flux;

namespace {

void test_sum_all() {
    tensor::Tensor<float> t(tensor::Shape({5}), {1, 2, 3, 4, 5});
    CHECK_EQ(ops::sum(t), 15.f);
}

void test_sum_axis0_2d() {
    // shape {3,2} 行主序 [1,2, 3,4, 5,6]，按 axis0 求和 -> [9, 12]
    tensor::Tensor<float> t(tensor::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    auto r = ops::sum(t, 0);
    CHECK(r.shape() == tensor::Shape({2}));
    CHECK_EQ(r[0], 9.f);
    CHECK_EQ(r[1], 12.f);
}

void test_sum_axis1_2d() {
    // 按 axis1 求和 -> [3, 7, 11]
    tensor::Tensor<float> t(tensor::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    auto r = ops::sum(t, 1);
    CHECK(r.shape() == tensor::Shape({3}));
    CHECK_EQ(r[0], 3.f);
    CHECK_EQ(r[1], 7.f);
    CHECK_EQ(r[2], 11.f);
}

void test_sum_axis_out_of_range() {
    tensor::Tensor<float> t(tensor::Shape({2, 2}), {1, 2, 3, 4});
    bool threw = false;
    try {
        (void)ops::sum(t, 5);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

void test_sum_nan_skip() {
    // pandas 风格 skipna：NaN 不参与求和
    tensor::Tensor<float> t(tensor::Shape({4}),
                            {1, std::numeric_limits<float>::quiet_NaN(), 3, 4});
    CHECK_EQ(ops::sum(t), 8.f);
}

void test_sum_all_nan() {
    // 全 NaN -> 0（pandas 对空有效集求和为 0）
    tensor::Tensor<float> t(tensor::Shape({2}), {std::numeric_limits<float>::quiet_NaN(),
                                                 std::numeric_limits<float>::quiet_NaN()});
    CHECK_EQ(ops::sum(t), 0.f);
}

void test_sum_nan_skip_axis() {
    tensor::Tensor<float> t(tensor::Shape({2, 2}),
                            {1, std::numeric_limits<float>::quiet_NaN(), 3, 4});
    auto r = ops::sum(t, 1);
    CHECK_EQ(r[0], 1.f);
    CHECK_EQ(r[1], 7.f);
}

void test_sum_int() {
    // 整数路径：std::isnan(int) 恒 false，全部累加
    tensor::Tensor<int> t(tensor::Shape({3}), {1, 2, 3});
    CHECK_EQ(ops::sum(t), 6);
}

} // namespace

int main() {
    test_sum_all();
    test_sum_axis0_2d();
    test_sum_axis1_2d();
    test_sum_axis_out_of_range();
    test_sum_nan_skip();
    test_sum_all_nan();
    test_sum_nan_skip_axis();
    test_sum_int();
    return flux_test::summary();
}
