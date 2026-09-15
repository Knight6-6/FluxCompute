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

    // 池内保留字节的上限，超出的归还给 OS 而不是继续攒着。
    //
    // 为什么需要这个上限：池按字节尺寸归档，尺寸固定时占用会收敛，
    // 但**尺寸从不重复**的负载（例如"筛选后标的数每天不同"）会让每个
    // 新尺寸都留下一个永远不会被复用的块。实测那种场景下 RSS 线性增长、
    // 不收敛（每轮约 2.3 MB），对长期运行的进程是个隐患。
    //
    // 256 MiB 对单机量化负载足够覆盖工作集；需要更大可调 set_max_pooled_bytes。
    static constexpr std::size_t kDefaultMaxPooledBytes = std::size_t(256) << 20;

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
            pooled_bytes_ -= bytes;
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
            // 超过上限就还给 OS。这样池的占用有界，不会因尺寸多变而无界增长。
            if (pooled_bytes_ + bytes > max_pooled_bytes_) {
                release_to_os(ptr);
                return;
            }
            free_blocks_[bytes].push_back(ptr);   // 这里插入是对的：确实要记这个尺寸
            pooled_bytes_ += bytes;
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

    // 观测用：池内当前保留的字节数。
    std::size_t pooled_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pooled_bytes_;
    }

    // 调整保留上限。设得比当前占用小不会立刻释放，只是此后归还的块不再入池，
    // 占用会随复用自然降下来。
    void set_max_pooled_bytes(std::size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        max_pooled_bytes_ = bytes;
    }

    std::size_t max_pooled_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return max_pooled_bytes_;
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
    std::size_t pooled_bytes_ = 0;                            // 各空闲链上的字节总和
    std::size_t max_pooled_bytes_ = kDefaultMaxPooledBytes;
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
