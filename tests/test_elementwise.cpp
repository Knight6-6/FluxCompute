#include <flux/flux.hpp>
#include "test_util.hpp"

#include <stdexcept>

using namespace flux;

namespace {

void test_add_1d() {
    tensor::Tensor<float> a(tensor::Shape({4}), {1, 2, 3, 4});
    tensor::Tensor<float> b(tensor::Shape({4}), {10, 20, 30, 40});
    auto c = ops::add(a, b);
    CHECK(c.shape() == tensor::Shape({4}));
    CHECK_EQ(c[0], 11.f);
    CHECK_EQ(c[1], 22.f);
    CHECK_EQ(c[2], 33.f);
    CHECK_EQ(c[3], 44.f);
}

void test_sub_mul_div() {
    tensor::Tensor<double> a(tensor::Shape({3}), {10, 20, 30});
    tensor::Tensor<double> b(tensor::Shape({3}), {2, 5, 4});
    CHECK_EQ(ops::sub(a, b)[0], 8.0);
    CHECK_EQ(ops::mul(a, b)[1], 100.0);
    CHECK_EQ(ops::div(a, b)[2], 7.5);
}

void test_2d() {
    tensor::Tensor<float> a(tensor::Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    tensor::Tensor<float> b(tensor::Shape({2, 3}), {1, 1, 1, 1, 1, 1});
    auto c = ops::add(a, b);
    CHECK_EQ(c[3], 5.f);
    CHECK_EQ(c[5], 7.f);
}

void test_shape_mismatch_throws() {
    tensor::Tensor<float> a(tensor::Shape({3}), {1, 2, 3});
    tensor::Tensor<float> b(tensor::Shape({2}), {1, 2});
    bool threw = false;
    try {
        (void)ops::add(a, b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

} // namespace

int main() {
    test_add_1d();
    test_sub_mul_div();
    test_2d();
    test_shape_mismatch_throws();
    return flux_test::summary();
}
