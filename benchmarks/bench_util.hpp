#pragma once

// FluxCompute 基准测试的极简框架：单头文件、无外部依赖，与 tests/test_util.hpp 同源。
//
// 三个刻意的取舍：
//
//   1. **取最小值而非均值。** 微基准里的噪声（调度、频率抖动、其他进程）
//      只会让某次测量变慢，不会变快，所以最小值最接近"没有干扰时的真实成本"。
//      均值反而把噪声混进了信号。
//
//   2. **每组先跑一次 warmup。** 首次触碰内存要建页表、填缓存，第一次的数字
//      系统性偏慢，不该计入。
//
//   3. **用 keep() 阻止死代码消除。** -O3 下编译器很可能把整段计算连同分配
//      一起优化掉，测出来接近零耗时——那种数字比不测更糟，因为它看起来很好。

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace flux_bench {

inline volatile const void* g_sink = nullptr;

// 把指针写进一个 volatile 全局：告诉编译器"这个地址逃逸了，别把它的计算删掉"。
inline void keep(const void* p) { g_sink = p; }

struct Entry {
    std::string name;
    double ms;
};

inline std::vector<Entry>& entries() {
    static std::vector<Entry> e;
    return e;
}

template <typename F>
inline double time_once_ms(F& body) {
    const auto t0 = std::chrono::steady_clock::now();
    body();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// reps 次取最小，只返回不记录。需要对两个实现做比值时用它。
// body 需自行用 keep() 保住结果。
template <typename F>
double best_ms(std::size_t reps, F body) {
    body();   // warmup，不计入

    double best = 1e300;
    for (std::size_t i = 0; i < reps; ++i) {
        const double ms = time_once_ms(body);
        if (ms < best) best = ms;
    }
    return best;
}

template <typename F>
void bench(const std::string& name, std::size_t reps, F body) {
    entries().push_back({name, best_ms(reps, body)});
}

inline int report(const char* title) {
    std::printf("\n=== %s ===\n", title);
    for (const auto& e : entries()) {
        std::printf("  %-48s %10.3f ms\n", e.name.c_str(), e.ms);
    }
    std::printf("  （每组取 %s，数值越小越快）\n\n", "多次重复的最小值");
    return 0;
}

}
