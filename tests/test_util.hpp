#pragma once

// FluxCompute 极简测试框架：单个头文件，无外部依赖。
//
// 用法：
//   在测试函数里用 CHECK / CHECK_EQ / CHECK_NEAR 断言；
//   main 返回 flux_test::summary()（失败时非零，供 CTest 判断）。

#include <iostream>
#include <cmath>

namespace flux_test {

struct Context {
    std::size_t checks = 0;
    std::size_t failures = 0;
};

inline Context& ctx() {
    static Context c;
    return c;
}

inline void check(bool ok, const char* expr, const char* file, int line) {
    ctx().checks++;
    if (!ok) {
        ctx().failures++;
        std::cerr << "FAIL " << file << ":" << line << "  " << expr << "\n";
    }
}

template <typename A, typename B>
void check_eq(const A& a, const B& b, const char* ea, const char* eb,
              const char* file, int line) {
    ctx().checks++;
    if (!(a == b)) {
        ctx().failures++;
        std::cerr << "FAIL " << file << ":" << line << "  " << ea << " == " << eb
                  << "  (got " << a << ", expected " << b << ")\n";
    }
}

template <typename T>
void check_near(T a, T b, T eps, const char* ea, const char* eb,
                const char* file, int line) {
    ctx().checks++;
    if (!(std::fabs(a - b) <= eps)) {
        ctx().failures++;
        std::cerr << "FAIL " << file << ":" << line << "  " << ea << " ≈ " << eb
                  << "  (got " << a << ", expected " << b << ")\n";
    }
}

inline int summary() {
    const auto& c = ctx();
    std::cout << "  " << c.checks << " checks, " << c.failures << " failures\n";
    return c.failures == 0 ? 0 : 1;
}

} // namespace flux_test

#define CHECK(expr)          flux_test::check(static_cast<bool>(expr), #expr, __FILE__, __LINE__)
#define CHECK_EQ(a, b)       flux_test::check_eq((a), (b), #a, #b, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, e)  flux_test::check_near((a), (b), (e), #a, #b, __FILE__, __LINE__)
