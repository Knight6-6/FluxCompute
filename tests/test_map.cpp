#include <flux/flux.hpp>
#include "test_util.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

using namespace flux;

namespace {

bool same(const tensor::Tensor<float>& a, const tensor::Tensor<float>& b) {
    if (a.shape() != b.shape()) return false;
    for (std::size_t i = 0; i < a.numel(); ++i) {
        const float x = a[i], y = b[i];
        if (std::isnan(x) && std::isnan(y)) continue;
        if (x != y) return false;
    }
    return true;
}

void test_map2_basic() {
    tensor::Tensor<float> a(tensor::Shape({4}), {1, 2, 3, 4});
    tensor::Tensor<float> b(tensor::Shape({4}), {10, 20, 30, 40});

    auto r = ops::map2(a, b, [](float x, float y) { return x + y; });
    CHECK(r.shape() == tensor::Shape({4}));
    CHECK_EQ(r[0], 11.f);
    CHECK_EQ(r[3], 44.f);
}

void test_map3_basic() {
    tensor::Tensor<float> a(tensor::Shape({3}), {1, 2, 3});
    tensor::Tensor<float> b(tensor::Shape({3}), {10, 20, 30});
    tensor::Tensor<float> c(tensor::Shape({3}), {2, 2, 2});

    // (a + b) * c
    auto r = ops::map3(a, b, c, [](float x, float y, float z) { return (x + y) * z; });
    CHECK_EQ(r[0], 22.f);
    CHECK_EQ(r[1], 44.f);
    CHECK_EQ(r[2], 66.f);
}

void test_map4_basic() {
    tensor::Tensor<float> a(tensor::Shape({3}), {1, 2, 3});
    tensor::Tensor<float> b(tensor::Shape({3}), {10, 20, 30});
    tensor::Tensor<float> c(tensor::Shape({3}), {2, 2, 2});
    tensor::Tensor<float> d(tensor::Shape({3}), {1, 1, 1});

    // (a + b) * c - d
    auto r = ops::map4(a, b, c, d, [](float x, float y, float z, float w) {
        return (x + y) * z - w;
    });
    CHECK_EQ(r[0], 21.f);
    CHECK_EQ(r[1], 43.f);
    CHECK_EQ(r[2], 65.f);
}

void test_map_matches_unfused_composition() {
    // 融合写法与分离写法的**结果必须完全一致**——这是融合的正确性底线。
    // 融合改变的是遍历次数，不是语义。
    tensor::Tensor<float> a(tensor::Shape({16}));
    tensor::Tensor<float> b(tensor::Shape({16}));
    tensor::Tensor<float> c(tensor::Shape({16}));
    for (std::size_t i = 0; i < 16; ++i) {
        a[i] = static_cast<float>(i) * 0.5f - 3.f;
        b[i] = static_cast<float>(i % 5) + 1.f;
        c[i] = static_cast<float>(i % 3) * 2.f - 1.f;
    }

    auto fused   = ops::map3(a, b, c, [](float x, float y, float z) { return (x + y) * z; });
    auto unfused = ops::mul(ops::add(a, b), c);

    CHECK(same(fused, unfused));
}

void test_map_shape_mismatch_throws() {
    tensor::Tensor<float> a(tensor::Shape({4}), {1, 2, 3, 4});
    tensor::Tensor<float> b(tensor::Shape({3}), {1, 2, 3});

    bool threw = false;
    try {
        (void)ops::map2(a, b, [](float x, float y) { return x + y; });
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_map3_shape_mismatch_throws() {
    tensor::Tensor<float> a(tensor::Shape({4}), {1, 2, 3, 4});
    tensor::Tensor<float> b(tensor::Shape({4}), {1, 2, 3, 4});
    tensor::Tensor<float> c(tensor::Shape({3}), {1, 2, 3});

    bool threw = false;
    try {
        (void)ops::map3(a, b, c, [](float x, float y, float z) { return x + y + z; });
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_map_does_not_skip_nan() {
    // 与 reduce / scan 同一约定：lambda 是任意标量函数，库无从知道它该怎么
    // 处理 NaN，故一律照常参与。NaN 会按 lambda 的语义传播出去。
    tensor::Tensor<float> a(tensor::Shape({3}),
                            {1, std::numeric_limits<float>::quiet_NaN(), 3});
    tensor::Tensor<float> b(tensor::Shape({3}), {1, 1, 1});

    auto r = ops::map2(a, b, [](float x, float y) { return x + y; });
    CHECK_EQ(r[0], 2.f);
    CHECK(std::isnan(r[1]));
    CHECK_EQ(r[2], 4.f);
}

void test_map_int() {
    tensor::Tensor<int> a(tensor::Shape({3}), {1, 2, 3});
    tensor::Tensor<int> b(tensor::Shape({3}), {10, 20, 30});

    auto r = ops::map2(a, b, [](int x, int y) { return x * y; });
    CHECK_EQ(r[0], 10);
    CHECK_EQ(r[2], 90);
}

void test_map_accepts_capturing_functor() {
    tensor::Tensor<float> a(tensor::Shape({3}), {1, 2, 3});
    tensor::Tensor<float> b(tensor::Shape({3}), {1, 1, 1});
    const float scale = 10.f;

    auto r = ops::map2(a, b, [scale](float x, float y) { return (x + y) * scale; });
    CHECK_EQ(r[0], 20.f);
    CHECK_EQ(r[2], 40.f);
}

void test_map_multidim() {
    // 多维按扁平顺序逐元素合并；融合不改变形状语义
    tensor::Tensor<float> a(tensor::Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    tensor::Tensor<float> b(tensor::Shape({2, 3}), {10, 10, 10, 20, 20, 20});

    auto r = ops::map2(a, b, [](float x, float y) { return x + y; });
    CHECK(r.shape() == tensor::Shape({2, 3}));
    CHECK_EQ(r[0], 11.f);
    CHECK_EQ(r[5], 26.f);
}

void test_map_empty_tensor() {
    // 零长度张量是合法的（见 test_tensor），融合路径也要能处理
    tensor::Tensor<float> a(tensor::Shape({0}));
    tensor::Tensor<float> b(tensor::Shape({0}));

    auto r = ops::map2(a, b, [](float x, float y) { return x + y; });
    CHECK_EQ(r.numel(), static_cast<std::size_t>(0));
}

} // namespace

int main() {
    test_map2_basic();
    test_map3_basic();
    test_map4_basic();
    test_map_matches_unfused_composition();
    test_map_shape_mismatch_throws();
    test_map3_shape_mismatch_throws();
    test_map_does_not_skip_nan();
    test_map_int();
    test_map_accepts_capturing_functor();
    test_map_multidim();
    test_map_empty_tensor();
    return flux_test::summary();
}
