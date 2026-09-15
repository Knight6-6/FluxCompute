#include <flux/flux.hpp>
#include "test_util.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace flux;

namespace {

void test_max_all() {
    tensor::Tensor<float> t(tensor::Shape({5}), {3, 7, 2, 9, 5});
    CHECK_EQ(ops::max(t), 9.f);
}

void test_max_axis0_2d() {
    tensor::Tensor<float> t(tensor::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    auto r = ops::max(t, 0);
    CHECK(r.shape() == tensor::Shape({2}));
    CHECK_EQ(r[0], 5.f);
    CHECK_EQ(r[1], 6.f);
}

void test_max_axis1_2d() {
    tensor::Tensor<float> t(tensor::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
    auto r = ops::max(t, 1);
    CHECK(r.shape() == tensor::Shape({3}));
    CHECK_EQ(r[0], 2.f);
    CHECK_EQ(r[1], 4.f);
    CHECK_EQ(r[2], 6.f);
}

void test_max_axis0_3d() {
    tensor::Tensor<float> t(tensor::Shape({2, 2, 3}), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto r = ops::max(t, 0);
    CHECK(r.shape() == tensor::Shape({2, 3}));
    for (std::size_t i = 0; i < 6; ++i) CHECK_EQ(r[i], 7.f + i);
}

void test_max_nan_skip() {
    tensor::Tensor<float> t(tensor::Shape({5}), {1, std::numeric_limits<float>::quiet_NaN(),
                                                 3, std::numeric_limits<float>::quiet_NaN(), 2});
    CHECK_EQ(ops::max(t), 3.f);
}

void test_max_all_nan() {
    tensor::Tensor<float> t(tensor::Shape({2}), {std::numeric_limits<float>::quiet_NaN(),
                                                 std::numeric_limits<float>::quiet_NaN()});
    CHECK(std::isnan(ops::max(t)));
}

void test_max_all_nan_axis() {
    tensor::Tensor<float> t(tensor::Shape({2, 2}),
                            {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             1, 2});
    auto r = ops::max(t, 1);
    CHECK(std::isnan(r[0]));
    CHECK_EQ(r[1], 2.f);
}

void test_max_int_negatives() {
    // 回归：全负数 int 切片，不能用 NaN 哨兵=0 兜底
    tensor::Tensor<int> t(tensor::Shape({4}), {-5, -1, -3, -2});
    CHECK_EQ(ops::max(t), -1);
}

void test_max_axis_int_negatives() {
    tensor::Tensor<int> t(tensor::Shape({2, 2}), {-5, -1, -3, -2});
    auto r = ops::max(t, 1);
    CHECK_EQ(r[0], -1);
    CHECK_EQ(r[1], -2);
}

void test_max_axis_out_of_range() {
    tensor::Tensor<float> t(tensor::Shape({2, 2}), {1, 2, 3, 4});
    bool threw = false;
    try {
        (void)ops::max(t, 5);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    test_max_all();
    test_max_axis0_2d();
    test_max_axis1_2d();
    test_max_axis0_3d();
    test_max_nan_skip();
    test_max_all_nan();
    test_max_all_nan_axis();
    test_max_int_negatives();
    test_max_axis_int_negatives();
    test_max_axis_out_of_range();
    return flux_test::summary();
}
