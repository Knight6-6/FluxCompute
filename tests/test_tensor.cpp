#include <flux/flux.hpp>
#include "test_util.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace flux;

namespace {

void test_shape_numel_and_strides() {
    // 行主序（C-order）：{2,3,4} 的 stride 应是 {12,4,1}
    tensor::Shape s({2, 3, 4});
    CHECK_EQ(s.numel(), static_cast<std::size_t>(24));
    CHECK_EQ(s.stride(0), static_cast<std::size_t>(12));
    CHECK_EQ(s.stride(1), static_cast<std::size_t>(4));
    CHECK_EQ(s.stride(2), static_cast<std::size_t>(1));
}

void test_shape_equality() {
    CHECK(tensor::Shape({2, 3}) == tensor::Shape({2, 3}));
    CHECK(tensor::Shape({2, 3}) != tensor::Shape({3, 2}));
}

void test_shape_zero_dim_does_not_crash() {
    // 回归：构造期算 stride 时做了 stride /= dimensions_[i]，任一维为 0 就是
    // 整数除零（0/0），直接 SIGFPE 杀掉进程——不是返回错误，是崩。
    // 零长度维度在真实数据里很常见（空交易日、没有标的通过筛选）。
    tensor::Shape empty({0});
    CHECK_EQ(empty.numel(), static_cast<std::size_t>(0));

    // 零出现在中间维，后面的维仍应能算出 stride
    tensor::Shape mixed({2, 0, 3});
    CHECK_EQ(mixed.numel(), static_cast<std::size_t>(0));
    CHECK_EQ(mixed.stride(0), static_cast<std::size_t>(0));
    CHECK_EQ(mixed.stride(1), static_cast<std::size_t>(0));
    CHECK_EQ(mixed.stride(2), static_cast<std::size_t>(0));
}

void test_tensor_zero_dim() {
    tensor::Tensor<float> t(tensor::Shape({0}));
    CHECK_EQ(t.numel(), static_cast<std::size_t>(0));
    CHECK(t.shape() == tensor::Shape({0}));
}

void test_tensor_element_access() {
    tensor::Tensor<float> t(tensor::Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    CHECK_EQ(t.numel(), static_cast<std::size_t>(6));
    CHECK_EQ(t[0], 1.f);
    CHECK_EQ(t[5], 6.f);

    t[2] = 99.f;   // 非 const 版本可写
    CHECK_EQ(t[2], 99.f);

    const auto& ct = t;
    CHECK_EQ(ct[2], 99.f);
}

void test_tensor_size_mismatch_throws() {
    bool threw = false;
    try {
        tensor::Tensor<float> t(tensor::Shape({3}), {1, 2});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_tensor_is_64_byte_aligned() {
    tensor::Tensor<float> t(tensor::Shape({100}));
    CHECK_EQ(reinterpret_cast<std::uintptr_t>(t.data()) % 64, static_cast<std::uintptr_t>(0));
}

} // namespace

int main() {
    test_shape_numel_and_strides();
    test_shape_equality();
    test_shape_zero_dim_does_not_crash();
    test_tensor_zero_dim();
    test_tensor_element_access();
    test_tensor_size_mismatch_throws();
    test_tensor_is_64_byte_aligned();
    return flux_test::summary();
}
