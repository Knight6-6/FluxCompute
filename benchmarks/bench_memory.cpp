// 内存池到底值不值得启用？
//
// 背景：#3 把 MemoryPool 修对了、做了个合规的 PooledAllocator，但**默认没启用**
// ——因为当时没有数据支撑"池化更快"这个假设，而且 README 原则 #4 要求
// 不在没有 Benchmark 的情况下盲目优化。这就是那份数据。
//
// ---------------------------------------------------------------------------
// 一个测量上的坑（第一版踩过，记在这里免得再踩）：
//
// 用 `std::vector<float, Alloc> v(elems);` 来制造分配，编译器会把**整段
// 优化掉**——缓冲区的内容没人观察，分配和清零都可以被合法消除。第一版就是
// 这么写的，测出"1MB 直连分配 200k 次只要 0.127ms"（25ns/次），物理上不可能。
//
// 所以这里直接调 allocate/deallocate，并触碰首尾元素，确保内存真的被申请、
// 也真的能访问。测的是分配器本身，不掺 std::vector 的初始化开销。
// ---------------------------------------------------------------------------

#include <flux/flux.hpp>
#include "bench_util.hpp"

#include <cstdio>

using namespace flux;

namespace {

using Direct = memory::AlignedAllocator<float, 64>;
using Pooled = memory::PooledAllocator<float>;

// 同尺寸反复申请/释放——内存池的理想场景
template <typename Alloc>
void churn(std::size_t rounds, std::size_t elems) {
    Alloc alloc;
    for (std::size_t i = 0; i < rounds; ++i) {
        float* p = alloc.allocate(elems);
        p[0] = 1.0f;
        p[elems - 1] = 2.0f;
        flux_bench::keep(p);           // 指针逃逸，分配不可被消除
        alloc.deallocate(p, elems);
    }
}

// 批量申请、整批释放——一次图执行的形态（一批中间张量同时存活）
template <typename Alloc>
void batch(std::size_t batches, std::size_t per_batch, std::size_t elems) {
    Alloc alloc;
    for (std::size_t b = 0; b < batches; ++b) {
        std::vector<float*> ps;
        ps.reserve(per_batch);
        for (std::size_t i = 0; i < per_batch; ++i) {
            float* p = alloc.allocate(elems);
            p[0] = 1.0f;
            ps.push_back(p);
        }
        flux_bench::keep(ps.back());
        for (float* p : ps) alloc.deallocate(p, elems);
    }
}

} // namespace

int main() {
    std::printf("对比对象是 glibc malloc（自带 tcache，同尺寸申请/释放本就很快）\n");

    flux_bench::bench("同尺寸 churn：直连分配 200k 轮 x 4KB", 20, [] {
        churn<Direct>(200000, 1024);
    });
    flux_bench::bench("同尺寸 churn：内存池   200k 轮 x 4KB", 20, [] {
        churn<Pooled>(200000, 1024);
    });

    flux_bench::bench("同尺寸 churn：直连分配 20k 轮 x 1MB", 20, [] {
        churn<Direct>(20000, 262144);
    });
    flux_bench::bench("同尺寸 churn：内存池   20k 轮 x 1MB", 20, [] {
        churn<Pooled>(20000, 262144);
    });

    flux_bench::bench("批量 churn：直连分配 2k 批 x 16 个 x 4KB", 20, [] {
        batch<Direct>(2000, 16, 1024);
    });
    flux_bench::bench("批量 churn：内存池   2k 批 x 16 个 x 4KB", 20, [] {
        batch<Pooled>(2000, 16, 1024);
    });

    return flux_bench::report("memory");
}
