#pragma once
#include <cstdlib>
#include <new>
#include <atomic>
#include <cstddef>

namespace flux::memory {

struct AllocationStats {
    std::size_t calls = 0;  // allocate 被调用的次数
    std::size_t bytes = 0;  // 累计申请的字节数
};

// 分配探针：统计经 AlignedAllocator 走出的所有分配。
//
// 之所以能当作"物化出的张量缓冲区个数"用：Tensor 唯一的堆分配就是它的
// ContainerType data_（Shape 走默认分配器，unordered_map 的节点与
// make_shared 的控制块同样走默认分配器），而 AlignedAllocator 目前只被
// Tensor 使用。因此这个计数是精确的，不是近似。
//
// 用途：断言"执行期没有发生多余的张量拷贝"——这是不依赖 benchmark
// 就能验证零拷贝的唯一精确手段。计数常开，relaxed 原子自增相对于一次
// 整块内存拷贝是噪声级开销。
//
// 故意做成非模板：计数器跨 T 共用一份，测试里不必区分元素类型。
class AllocationCounter {
public:
    static AllocationStats snapshot() {
        return {calls_.load(std::memory_order_relaxed),
                bytes_.load(std::memory_order_relaxed)};
    }

    static void record(std::size_t bytes) {
        calls_.fetch_add(1, std::memory_order_relaxed);
        bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }

private:
    static inline std::atomic<std::size_t> calls_{0};
    static inline std::atomic<std::size_t> bytes_{0};
};

template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() noexcept = default;
    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            ptr = nullptr;
        }
#endif
        if (!ptr) throw std::bad_alloc();
        // 记在成功的分支上：空张量（n == 0）在上面已提前返回，不计入。
        AllocationCounter::record(n * sizeof(T));
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        if (!p) return;
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    template <typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept { return false; }
};

}