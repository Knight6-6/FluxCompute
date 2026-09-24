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

void test_rank_large_scale() {
    // 仿真真实 A 股 26 窗口 x 7370 标的
    const std::size_t T = 26;
    const std::size_t N = 7370;
    tensor::Tensor<float> tensor(tensor::Shape({T, N}));
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t n = 0; n < N; ++n) {
            if (n % 10 == 0) {
                tensor[t * N + n] = std::numeric_limits<float>::quiet_NaN();
            } else if (n % 5 == 0) {
                tensor[t * N + n] = 100.0f; // 大量并列
            } else {
                tensor[t * N + n] = static_cast<float>((n * 17 + t * 31) % 10000);
            }
        }
    }
    auto r = ops::rank(tensor, 1, true); // 横截面百分比排序
    CHECK(r.shape() == tensor::Shape({T, N}));

    for (std::size_t t = 0; t < T; ++t) {
        double sum = 0.0;
        std::size_t valid = 0;
        for (std::size_t n = 0; n < N; ++n) {
            float v = r[t * N + n];
            if (n % 10 == 0) {
                CHECK(std::isnan(v));
            } else {
                CHECK(!std::isnan(v));
                CHECK(v > 0.0f && v <= 1.0f);
                sum += v;
                valid++;
            }
        }
        CHECK(valid == N - (N + 9) / 10);
        double mean = sum / valid;
        // 均匀分布百分比均值严格接近 0.5
        CHECK(std::abs(mean - 0.5) < 0.05);
    }
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
    test_rank_large_scale();
    test_rank_axis_out_of_range();
    return flux_test::summary();
}
