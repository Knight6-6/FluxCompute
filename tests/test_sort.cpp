#include <flux/flux.hpp>
#include "test_util.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace flux;

namespace {

void test_sort_1d_asc() {
    tensor::Tensor<float> t(tensor::Shape({4}), {10, 3, 7, 1});
    auto r = ops::sort(t);
    CHECK_EQ(r[0], 1.f);
    CHECK_EQ(r[1], 3.f);
    CHECK_EQ(r[2], 7.f);
    CHECK_EQ(r[3], 10.f);
}

void test_sort_1d_desc() {
    tensor::Tensor<float> t(tensor::Shape({4}), {10, 3, 7, 1});
    auto r = ops::sort(t, 0, false);
    CHECK_EQ(r[0], 10.f);
    CHECK_EQ(r[1], 7.f);
    CHECK_EQ(r[2], 3.f);
    CHECK_EQ(r[3], 1.f);
}

void test_sort_1d_nan_last_asc() {
    // NaN 恒排最后（升序）
    tensor::Tensor<float> t(tensor::Shape({4}), {3, std::numeric_limits<float>::quiet_NaN(), 1, 2});
    auto r = ops::sort(t);
    CHECK_EQ(r[0], 1.f);
    CHECK_EQ(r[1], 2.f);
    CHECK_EQ(r[2], 3.f);
    CHECK(std::isnan(r[3]));
}

void test_sort_1d_nan_last_desc() {
    // NaN 依旧排最后（降序）
    tensor::Tensor<float> t(tensor::Shape({4}), {3, std::numeric_limits<float>::quiet_NaN(), 1, 2});
    auto r = ops::sort(t, 0, false);
    CHECK_EQ(r[0], 3.f);
    CHECK_EQ(r[1], 2.f);
    CHECK_EQ(r[2], 1.f);
    CHECK(std::isnan(r[3]));
}

void test_sort_2d_axis0() {
    // shape {2,3}: row0=[5,2,8], row1=[1,9,3]，按列排序 -> [1,2,3,5,9,8]
    tensor::Tensor<float> t(tensor::Shape({2, 3}), {5, 2, 8, 1, 9, 3});
    auto r = ops::sort(t, 0);
    CHECK(r.shape() == tensor::Shape({2, 3}));
    CHECK_EQ(r[0], 1.f);
    CHECK_EQ(r[1], 2.f);
    CHECK_EQ(r[2], 3.f);
    CHECK_EQ(r[3], 5.f);
    CHECK_EQ(r[4], 9.f);
    CHECK_EQ(r[5], 8.f);
}

void test_sort_2d_axis1() {
    // 按行排序：row0 -> [2,5,8], row1 -> [1,3,9]
    tensor::Tensor<float> t(tensor::Shape({2, 3}), {5, 2, 8, 1, 9, 3});
    auto r = ops::sort(t, 1);
    CHECK(r.shape() == tensor::Shape({2, 3}));
    CHECK_EQ(r[0], 2.f);
    CHECK_EQ(r[1], 5.f);
    CHECK_EQ(r[2], 8.f);
    CHECK_EQ(r[3], 1.f);
    CHECK_EQ(r[4], 3.f);
    CHECK_EQ(r[5], 9.f);
}

void test_sort_int() {
    tensor::Tensor<int> t(tensor::Shape({3}), {3, 1, 2});
    auto r = ops::sort(t);
    CHECK_EQ(r[0], 1);
    CHECK_EQ(r[1], 2);
    CHECK_EQ(r[2], 3);
}

void test_sort_axis_out_of_range() {
    tensor::Tensor<float> t(tensor::Shape({2, 2}), {1, 2, 3, 4});
    bool threw = false;
    try {
        (void)ops::sort(t, 5);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

// 稳定性：值输出无法直接观察（并列元素值相同），由 std::stable_sort 保证，不作测试。

int main() {
    test_sort_1d_asc();
    test_sort_1d_desc();
    test_sort_1d_nan_last_asc();
    test_sort_1d_nan_last_desc();
    test_sort_2d_axis0();
    test_sort_2d_axis1();
    test_sort_int();
    test_sort_axis_out_of_range();
    return flux_test::summary();
}
