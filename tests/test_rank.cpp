#include <flux/flux.hpp>
#include "test_util.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace flux;

namespace {

void test_rank_1d_no_ties() {
    tensor::Tensor<float> t(tensor::Shape({3}), {10, 20, 30});
    auto r = ops::rank(t);
    CHECK_EQ(r[0], 1.f);
    CHECK_EQ(r[1], 2.f);
    CHECK_EQ(r[2], 3.f);
}

void test_rank_1d_ties() {
    // 并列取平均名次
    tensor::Tensor<float> t(tensor::Shape({4}), {10, 20, 20, 30});
    auto r = ops::rank(t);
    CHECK_EQ(r[0], 1.f);
    CHECK_EQ(r[1], 2.5f);
    CHECK_EQ(r[2], 2.5f);
    CHECK_EQ(r[3], 4.f);
}

void test_rank_nan_keep() {
    tensor::Tensor<float> t(tensor::Shape({3}), {10, std::numeric_limits<float>::quiet_NaN(), 30});
    auto r = ops::rank(t);
    CHECK_EQ(r[0], 1.f);
    CHECK(std::isnan(r[1]));
    CHECK_EQ(r[2], 2.f);
}

void test_rank_2d_axis0() {
    // shape {2,2}: col0=[10,5], col1=[20,25] -> [2,1,1,2]
    tensor::Tensor<float> t(tensor::Shape({2, 2}), {10, 20, 5, 25});
    auto r = ops::rank(t, 0);
    CHECK(r.shape() == tensor::Shape({2, 2}));
    CHECK_EQ(r[0], 2.f);  // 10 在 col0 中排第 2
    CHECK_EQ(r[1], 1.f);  // 20 在 col1 中排第 1
    CHECK_EQ(r[2], 1.f);  // 5 排第 1
    CHECK_EQ(r[3], 2.f);  // 25 排第 2
}

void test_rank_2d_axis1_ties() {
    // shape {2,3}: row0=[10,20,20] -> [1,2.5,2.5]; row1=[30,10,20] -> [3,1,2]
    tensor::Tensor<float> t(tensor::Shape({2, 3}), {10, 20, 20, 30, 10, 20});
    auto r = ops::rank(t, 1);
    CHECK(r.shape() == tensor::Shape({2, 3}));
    CHECK_EQ(r[0], 1.f);
    CHECK_EQ(r[1], 2.5f);
    CHECK_EQ(r[2], 2.5f);
    CHECK_EQ(r[3], 3.f);
    CHECK_EQ(r[4], 1.f);
    CHECK_EQ(r[5], 2.f);
}

void test_rank_pct() {
    tensor::Tensor<float> t(tensor::Shape({3}), {10, 20, 30});
    auto r = ops::rank(t, 0, true);
    CHECK_NEAR(r[0], 1.f / 3.f, 1e-6f);
    CHECK_NEAR(r[1], 2.f / 3.f, 1e-6f);
    CHECK_NEAR(r[2], 1.f, 1e-6f);
}

void test_rank_int_trunc() {
    // 整数 T：并列平均名次小数被截断（2.5 -> 2）
    tensor::Tensor<int> t(tensor::Shape({4}), {10, 20, 20, 30});
    auto r = ops::rank(t);
    CHECK_EQ(r[0], 1);
    CHECK_EQ(r[1], 2);
    CHECK_EQ(r[2], 2);
    CHECK_EQ(r[3], 4);
}

void test_rank_axis_out_of_range() {
    tensor::Tensor<float> t(tensor::Shape({2, 2}), {1, 2, 3, 4});
    bool threw = false;
    try {
        (void)ops::rank(t, 5);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    test_rank_1d_no_ties();
    test_rank_1d_ties();
    test_rank_nan_keep();
    test_rank_2d_axis0();
    test_rank_2d_axis1_ties();
    test_rank_pct();
    test_rank_int_trunc();
    test_rank_axis_out_of_range();
    return flux_test::summary();
}
