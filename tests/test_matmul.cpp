#include <flux/flux.hpp>
#include "test_util.hpp"

#include <stdexcept>
#include <vector>

using namespace flux;

namespace {

void test_matmul_2x2() {
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    tensor::Tensor<float> a(tensor::Shape({2, 2}), {1, 2, 3, 4});
    tensor::Tensor<float> b(tensor::Shape({2, 2}), {5, 6, 7, 8});

    auto c = ops::matmul(a, b);
    CHECK(c.shape() == tensor::Shape({2, 2}));
    CHECK_EQ(c[0], 19.f);
    CHECK_EQ(c[1], 22.f);
    CHECK_EQ(c[2], 43.f);
    CHECK_EQ(c[3], 50.f);
}

void test_matmul_non_square() {
    // (2,3) * (3,4) -> (2,4)
    tensor::Tensor<float> a(tensor::Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    tensor::Tensor<float> b(tensor::Shape({3, 4}),
                            {1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0});   // 前三列的 3x3 单位阵嵌在 3x4 里

    auto c = ops::matmul(a, b);
    CHECK(c.shape() == tensor::Shape({2, 4}));
    // C 的前三列等于 A，第四列全 0
    CHECK_EQ(c[0], 1.f);  CHECK_EQ(c[1], 2.f);  CHECK_EQ(c[2], 3.f);  CHECK_EQ(c[3], 0.f);
    CHECK_EQ(c[4], 4.f);  CHECK_EQ(c[5], 5.f);  CHECK_EQ(c[6], 6.f);  CHECK_EQ(c[7], 0.f);
}

void test_matmul_identity() {
    tensor::Tensor<float> a(tensor::Shape({3, 3}), {2, -1, 0, 5, 3, 7, 1, 1, 1});
    tensor::Tensor<float> i(tensor::Shape({3, 3}), {1, 0, 0, 0, 1, 0, 0, 0, 1});

    auto c = ops::matmul(a, i);
    for (std::size_t k = 0; k < 9; ++k) CHECK_EQ(c[k], a[k]);
}

void test_matmul_matches_naive_reference() {
    // 与教科书三重循环（i-j-k）逐元素对照。两者的循环序不同，结果必须一致
    // ——循环序只影响访存，不影响语义。
    const std::size_t m = 7, k = 5, n = 6;
    tensor::Tensor<float> a(tensor::Shape({m, k}));
    tensor::Tensor<float> b(tensor::Shape({k, n}));
    for (std::size_t i = 0; i < m * k; ++i) a[i] = static_cast<float>((i * 7) % 13) - 6.f;
    for (std::size_t i = 0; i < k * n; ++i) b[i] = static_cast<float>((i * 5) % 11) - 5.f;

    auto c = ops::matmul(a, b);

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float expect = 0.f;
            for (std::size_t p = 0; p < k; ++p) {
                expect += a[i * k + p] * b[p * n + j];
            }
            CHECK_NEAR(c[i * n + j], expect, 1e-3f);
        }
    }
}

void test_matmul_int() {
    tensor::Tensor<int> a(tensor::Shape({2, 2}), {1, 2, 3, 4});
    tensor::Tensor<int> b(tensor::Shape({2, 2}), {1, 0, 0, 1});

    auto c = ops::matmul(a, b);
    CHECK_EQ(c[0], 1);
    CHECK_EQ(c[3], 4);
}

void test_matmul_1x1() {
    tensor::Tensor<float> a(tensor::Shape({1, 1}), {3.f});
    tensor::Tensor<float> b(tensor::Shape({1, 1}), {4.f});
    CHECK_EQ(ops::matmul(a, b)[0], 12.f);
}

void test_matmul_zero_inner_dim() {
    // K = 0：内维为空，乘积约定为全 0。零长度维度是合法的（见 test_tensor），
    // 融合、归约各条路径都要能处理，matmul 也不例外。
    tensor::Tensor<float> a(tensor::Shape({2, 0}));
    tensor::Tensor<float> b(tensor::Shape({0, 3}));

    auto c = ops::matmul(a, b);
    CHECK(c.shape() == tensor::Shape({2, 3}));
    for (std::size_t i = 0; i < 6; ++i) CHECK_EQ(c[i], 0.f);
}

void test_matmul_requires_2d() {
    tensor::Tensor<float> a(tensor::Shape({4}), {1, 2, 3, 4});
    tensor::Tensor<float> b(tensor::Shape({2, 2}), {1, 2, 3, 4});

    bool threw = false;
    try {
        (void)ops::matmul(a, b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_matmul_inner_dim_mismatch_throws() {
    tensor::Tensor<float> a(tensor::Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    tensor::Tensor<float> b(tensor::Shape({4, 2}), {1, 2, 3, 4, 5, 6, 7, 8});

    bool threw = false;
    try {
        (void)ops::matmul(a, b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    test_matmul_2x2();
    test_matmul_non_square();
    test_matmul_identity();
    test_matmul_matches_naive_reference();
    test_matmul_int();
    test_matmul_1x1();
    test_matmul_zero_inner_dim();
    test_matmul_requires_2d();
    test_matmul_inner_dim_mismatch_throws();
    return flux_test::summary();
}
