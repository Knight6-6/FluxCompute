#pragma once
#include <flux/memory/aligned_allocator.hpp>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstddef>
#include <new>

namespace flux::memory {

// 布局：按字节尺寸（Size Class）组织空闲的对齐内存块。
//
// 定位：给"短生命周期、反复申请/释放同一批尺寸"的场景复用缓冲区，典型是
// 计算图执行期的中间张量。它是 Tensor 的默认分配器，理由见下方 PooledAllocator。
//
// 两个已知取舍，启用前需要知道：
//   1. 全局单例 + 一把 mutex。单线程下开销很低，但多线程并发分配会互相争抢；
//      若要用于按依赖层并行执行，得换成线程本地池或分片锁。
//   2. 回收的内存不还给 OS，直到进程退出才统一释放。这是池的常态，
//      代价是峰值占用会一直留着。
class MemoryPool {
public:
    static constexpr std::size_t kAlignment = 64;

    static MemoryPool& instance() {
        static MemoryPool pool;   // C++11 起函数局部静态的初始化是线程安全的
        return pool;
    }

    // 失败抛 std::bad_alloc，与 AlignedAllocator 保持一致。
    // 不用"失败返回 nullptr"那套：这个返回值太容易被调用方忽略，
    // 而在这里忽略一次就是空指针解引用。
    void* allocate(std::size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);

        // 用 find 而非 operator[]：后者在未命中时会插入一个空 vector，
        // 于是每出现一个新尺寸就在表里永久留一项，池子只涨不缩。
        auto it = free_blocks_.find(bytes);
        if (it != free_blocks_.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }

        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(bytes, kAlignment);
#else
        if (posix_memalign(&ptr, kAlignment, bytes) != 0) ptr = nullptr;
#endif
        if (!ptr) throw std::bad_alloc();
        return ptr;
    }

    void deallocate(void* ptr, std::size_t bytes) noexcept {
        if (!ptr) return;
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            free_blocks_[bytes].push_back(ptr);   // 这里插入是对的：确实要记这个尺寸
        } catch (...) {
            // deallocate 是 noexcept 路径（容器依赖这一点），池表扩容失败时
            // 不能抛。退回给 OS：宁可不复用，也不在释放路径上终止进程。
            release_to_os(ptr);
        }
    }

    // 观测用：当前记录了多少个尺寸档。测试用它盯住"未命中不留痕"。
    std::size_t size_classes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_blocks_.size();
    }

    ~MemoryPool() {
        for (auto& entry : free_blocks_) {
            for (void* ptr : entry.second) release_to_os(ptr);
        }
    }

private:
    static void release_to_os(void* ptr) noexcept {
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    mutable std::mutex mutex_;
    std::unordered_map<std::size_t, std::vector<void*>> free_blocks_;
};

// 走 MemoryPool 的分配器。**它就是 Tensor 的默认分配器。**
//
// 这个决定是 benchmark 逼出来的，不是性能洁癖：图执行每轮都新建/释放中间
// 张量，直连分配会落进 glibc 的 mmap/munmap 路径——每轮 2016 次缺页，占掉
// 约 75% 的执行时间。池化把缺页降到 0，扇出图 4.39 -> 0.99 ms/轮（4.4 倍）。
// 完整证据与推理见 docs/设计问题与取舍.md。
//
// 已知代价，改动它之前需要知道：
//   1. 全局单例 + 一把 mutex。单线程下开销很低，但多线程并发分配会互相争抢；
//      若要用于按依赖层并行执行，需要先换成分片锁或线程本地池。
//   2. 回收的内存不还给 OS，直到进程退出才统一释放，峰值占用会一直留着。
//   3. **静态析构顺序**：MemoryPool 是函数局部静态。若使用方在命名空间作用域
//      持有 Tensor，它的析构可能发生在池销毁之后，那次 deallocate 就是
//      use-after-destruction。真出现那种用法，得把单例改成"故意不析构"
//      （并相应处理 LeakSanitizer 的误报）。
//
// 与 AlignedAllocator 的差别仅在分配来源；对齐仍由 MemoryPool 固定为 64 字节，
// 所以只接受 Alignment <= 64 的请求。
template <typename T, std::size_t Alignment = 64>
struct PooledAllocator {
    static_assert(Alignment <= MemoryPool::kAlignment,
                  "MemoryPool 固定 64 字节对齐，无法满足更大的对齐要求");

    using value_type = T;

    PooledAllocator() noexcept = default;
    template <typename U>
    constexpr PooledAllocator(const PooledAllocator<U, Alignment>&) noexcept {}

    // 必须显式提供：Alignment 是非类型模板参数，默认的 rebind 推导不出来
    // （这个坑本项目在 AlignedAllocator 上踩过一次）。
    template <typename U>
    struct rebind {
        using other = PooledAllocator<U, Alignment>;
    };

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = MemoryPool::instance().allocate(n * sizeof(T));
        // 计数语义与 AlignedAllocator 对齐：记的是"发出去了一个缓冲区"，
        // 无论它来自池还是新申请。这样"run() 的缓冲区数 == 算子节点数"
        // 这条不变量在换用池化分配器后依然成立。
        AllocationCounter::record(n * sizeof(T));
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t n) noexcept {
        MemoryPool::instance().deallocate(p, n * sizeof(T));
    }

    template <typename U>
    bool operator==(const PooledAllocator<U, Alignment>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const PooledAllocator<U, Alignment>&) const noexcept { return false; }
};

}
