#include <flux/flux.hpp>
#include "test_util.hpp"

#include <algorithm>
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

void test_pool_cap_zero_disables_pooling() {
    // 上限为 0 时任何块都不入池，直接还给 OS——这是上限机制最直接的验证，
    // 且不受池的历史状态影响（断言的是"不增长"而非某个绝对值）。
    auto& pool = memory::MemoryPool::instance();
    const std::size_t saved = pool.max_pooled_bytes();

    pool.set_max_pooled_bytes(0);
    const std::size_t before = pool.pooled_bytes();

    void* p = pool.allocate(8192);
    pool.deallocate(p, 8192);
    CHECK_EQ(pool.pooled_bytes(), before);

    pool.set_max_pooled_bytes(saved);
}

void test_pool_bounded_with_never_repeating_sizes() {
    // 回归：池按字节尺寸归档，尺寸从不重复的负载会让每个新尺寸都留下一个
    // 永不复用的块。实测那种场景下 RSS 线性增长、不收敛（每轮约 2.3 MB）。
    // 上限保证占用有界。
    auto& pool = memory::MemoryPool::instance();
    const std::size_t saved = pool.max_pooled_bytes();
    const std::size_t cap = 128 * 1024;

    const std::size_t before = pool.pooled_bytes();
    pool.set_max_pooled_bytes(cap);

    for (int round = 0; round < 64; ++round) {
        // 每轮尺寸都不同，保证不复用任何尺寸档
        const std::size_t n = 1000 + static_cast<std::size_t>(round) * 37 + 1;
        tensor::Tensor<float> t(tensor::Shape({n}));
        t[0] = 1.f;
    }   // 析构时归还给池

    // 上限生效：占用不会超过 max(原有, 上限)
    CHECK(pool.pooled_bytes() <= std::max(before, cap));

    pool.set_max_pooled_bytes(saved);
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
    // 这两项会临时改上限，也放最后，避免影响其它用例的池行为
    test_pool_cap_zero_disables_pooling();
    test_pool_bounded_with_never_repeating_sizes();
    return flux_test::summary();
}
