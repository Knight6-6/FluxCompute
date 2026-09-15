#include <flux/flux.hpp>
#include "test_util.hpp"

#include <cstdint>
#include <limits>
#include <new>
#include <utility>
#include <vector>

using namespace flux;

namespace {

constexpr std::uintptr_t kAlign = memory::MemoryPool::kAlignment;

bool is_aligned(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p) % kAlign == 0;
}

void test_pool_reuses_block() {
    // 归还后再申请同尺寸，应当原样拿回同一块，不再向 OS 要。
    auto& pool = memory::MemoryPool::instance();
    void* a = pool.allocate(3333);
    pool.deallocate(a, 3333);
    void* b = pool.allocate(3333);
    CHECK(a == b);
    pool.deallocate(b, 3333);
}

void test_pool_alignment() {
    auto& pool = memory::MemoryPool::instance();
    const std::size_t sizes[] = {1, 64, 100, 4096, 100000};
    for (std::size_t bytes : sizes) {
        void* p = pool.allocate(bytes);
        CHECK(is_aligned(p));
        pool.deallocate(p, bytes);
    }
}

void test_pool_throws_on_failure() {
    // 回归：allocate 曾经在失败时静默返回 nullptr，与 AlignedAllocator 抛
    // bad_alloc 的行为不一致——而忽略一次返回值就是空指针解引用。
    auto& pool = memory::MemoryPool::instance();
    bool threw = false;
    try {
        (void)pool.allocate(std::numeric_limits<std::size_t>::max());
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    CHECK(threw);
}

void test_pool_miss_leaves_no_trace() {
    // 回归：allocate 曾用 operator[] 取桶，于是每出现一个新尺寸就在表里
    // 永久插入一个空 vector，池子只涨不缩。未命中必须不留痕。
    auto& pool = memory::MemoryPool::instance();
    const std::size_t before = pool.size_classes();

    std::vector<std::pair<void*, std::size_t>> held;
    for (std::size_t i = 1; i <= 16; ++i) {
        const std::size_t bytes = i * 4096 + 7;   // 一批此前没出现过的尺寸
        held.emplace_back(pool.allocate(bytes), bytes);
    }
    CHECK_EQ(pool.size_classes(), before);

    // 归还才建档，这是必要的：不记下来就没法复用。
    for (const auto& h : held) pool.deallocate(h.first, h.second);
    CHECK_EQ(pool.size_classes(), before + 16);
}

void test_pool_does_not_share_across_sizes() {
    auto& pool = memory::MemoryPool::instance();
    void* small = pool.allocate(128);
    pool.deallocate(small, 128);
    void* big = pool.allocate(4096);
    CHECK(small != big);   // 尺寸档之间不串用
    pool.deallocate(big, 4096);
}

void test_pooled_allocator_with_vector() {
    std::vector<int, memory::PooledAllocator<int>> v;
    for (int i = 0; i < 100; ++i) v.push_back(i * 3);

    CHECK_EQ(v.size(), static_cast<std::size_t>(100));
    CHECK_EQ(v[0], 0);
    CHECK_EQ(v[99], 297);
    CHECK(is_aligned(v.data()));
}

void test_pooled_allocator_records_allocations() {
    // 换用池化分配器后，计数仍然记"发出去了一个缓冲区"（无论来自池还是
    // 新申请），因此 #2 那条"run() 缓冲区数 == 算子节点数"的不变量依旧成立。
    std::vector<int, memory::PooledAllocator<int>> v;
    v.reserve(8);

    const auto before = memory::AllocationCounter::snapshot();
    v.reserve(64);   // 触发一次重新分配
    const auto after = memory::AllocationCounter::snapshot();

    CHECK_EQ(after.calls - before.calls, static_cast<std::size_t>(1));
}

void test_tensor_uses_pooled_allocator() {
    // Tensor 默认走池化分配器。这不是性能洁癖，是实测逼出来的决定——
    // 图执行每轮都新建/释放 4MB 中间张量，直连分配会落进 glibc 的
    // mmap/munmap 路径，每轮 2016 次缺页、占掉约 75% 的执行时间。
    // 池化后缺页归零，扇出图 4.39 -> 0.99 ms/轮（4.4 倍）。
    //
    // 证据与推理见 docs/设计问题与取舍.md。若有人把 Tensor 的分配器改回
    // AlignedAllocator，这条会立刻失败。
    //
    // 断言用"同一尺寸两次分配拿到同一地址"——这是复用的直接证据，
    // 不像池表计数那样会被其它测试的分配行为干扰。
    // 7919 是质数，避免与其它用例的尺寸撞档。
    void* first = nullptr;
    {
        tensor::Tensor<float> t(tensor::Shape({7919}));
        first = t.data();
    }
    {
        tensor::Tensor<float> t(tensor::Shape({7919}));
        CHECK(t.data() == first);
    }
}

} // namespace

int main() {
    test_pool_reuses_block();
    test_pool_alignment();
    test_pool_throws_on_failure();
    test_pool_miss_leaves_no_trace();
    test_pool_does_not_share_across_sizes();
    test_pooled_allocator_with_vector();
    test_pooled_allocator_records_allocations();
    test_tensor_uses_pooled_allocator();   // 放最后：它会给池表建档，别影响其它用例
    return flux_test::summary();
}
