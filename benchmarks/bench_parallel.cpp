// 并行调度能带来多少收益？（动手实现之前先量）
//
// 按依赖层并行的收益上限取决于两件事：
//   1. **图的宽度**——同一层里有多少个互不依赖的节点。宽度 1 就没得并行；
//   2. **内存带宽**——如果算子本身是内存受限的，多核一起跑会被带宽卡住，
//      加速比远低于核数。
//
// 这个基准把两者都量出来，用来判断"值不值得做"以及"上限在哪"。
//
// 另外单独量 MemoryPool 的全局锁：Tensor 的默认分配器是池化的，而池背后是
// 一把全局 mutex。多线程并发分配会不会先被这把锁卡住，是并行落地前的
// 前置问题。

#include <flux/flux.hpp>
#include "bench_util.hpp"

#include <atomic>
#include <cstdio>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

using namespace flux;

namespace {

constexpr std::size_t kElems = 1u << 20;   // 1M float = 4 MB

tensor::Tensor<float> make_data(std::size_t n) {
    tensor::Tensor<float> t(tensor::Shape({n}));
    for (std::size_t i = 0; i < n; ++i) t[i] = static_cast<float>(i % 97) * 0.5f + 1.0f;
    return t;
}

// 宽图：1 个输入喂 width 个互不依赖的算子，各自到一个输出。
// 同一层里有 width 个可并行的节点。
using OpFn = std::function<tensor::Tensor<float>(const tensor::Tensor<float>&)>;

struct WideGraph {
    graph::Graph<float> g;
    std::shared_ptr<graph::Node<float>> in;
    std::vector<std::shared_ptr<graph::Node<float>>> outs;

    WideGraph(std::size_t width, const OpFn& op) {
        in = g.create_node("in", graph::NodeType::Input);
        for (std::size_t i = 0; i < width; ++i) {
            auto n   = g.create_node("op", graph::NodeType::Operator);
            auto out = g.create_node("out", graph::NodeType::Output);
            g.add_edge(in, n);
            g.add_edge(n, out);
            g.bind_op(n, [op](const std::vector<tensor::TensorPtr<float>>& ins) {
                return op(*ins[0]);
            });
            outs.push_back(out);
        }
    }
};

// 池的并发分配测试：threads 个线程各做 rounds 次分配/释放
void pool_contention(std::size_t threads, std::size_t rounds, std::size_t elems) {
    std::atomic<std::size_t> ready{0};
    std::vector<std::thread> ts;
    ts.reserve(threads);

    for (std::size_t t = 0; t < threads; ++t) {
        ts.emplace_back([&, t] {
            memory::PooledAllocator<float> alloc;
            ++ready;
            while (ready.load() < threads) { /* 尽量同时起跑 */ }
            // 每个线程用自己的尺寸，避免互相复用同一批块而掩盖锁的开销
            const std::size_t n = elems + t;
            for (std::size_t i = 0; i < rounds; ++i) {
                float* p = alloc.allocate(n);
                p[0] = 1.0f;
                flux_bench::keep(p);
                alloc.deallocate(p, n);
            }
        });
    }
    for (auto& th : ts) th.join();
}

} // namespace

// 同一张宽图，串行 run() 与按层并行 run_parallel() 各测一次，给出加速比。
void compare(const char* label, std::size_t width, std::size_t elems,
             const OpFn& op, unsigned threads, int reps) {
    const auto data = make_data(elems);
    WideGraph wg(width, op);

    runtime::Executor<float> ex;
    ex.set_input(wg.in, data);

    const double seq = flux_bench::best_ms(reps, [&] {
        ex.run(wg.g);
        for (auto& o : wg.outs) flux_bench::keep(ex.get_output(o).get());
    });

    runtime::ThreadPool pool(threads);
    runtime::Executor<float> ex2;
    ex2.set_input(wg.in, data);

    const double par = flux_bench::best_ms(reps, [&] {
        ex2.run_parallel(wg.g, pool);
        for (auto& o : wg.outs) flux_bench::keep(ex2.get_output(o).get());
    });

    std::printf("  %-28s 宽 %2zu  串行 %8.3f ms   并行 %8.3f ms   加速比 %5.2fx\n",
                label, width, seq, par, seq / par);
}

int main() {
    const unsigned hw = std::thread::hardware_concurrency();
    std::printf("硬件线程数：%u\n", hw);
    std::printf("1M float = %.1f MB\n",
                static_cast<double>(kElems) * sizeof(float) / (1024 * 1024));

    auto mul_op  = [](const tensor::Tensor<float>& t) { return ops::mul(t, t); };
    auto sort_op = [](const tensor::Tensor<float>& t) { return ops::sort(t); };

    // --- 内存受限：逐元素 mul，1M float ---
    std::printf("\n-- 内存受限（逐元素 mul，1M float / 4MB，超出 L3）--\n");
    for (std::size_t width : {2u, 4u, 8u, 16u}) {
        compare("mul", width, kElems, mul_op, hw, 5);
    }

    // --- 计算受限：sort，256K float ---
    std::printf("\n-- 计算受限（sort，256K float / 1MB）--\n");
    for (std::size_t width : {2u, 4u, 8u, 16u}) {
        compare("sort", width, 262144, sort_op, hw, 3);
    }

    // --- 异步：参数搜索形态（多组任务同时在池里跑）---
    // 这是 run_async 的主要用途。注意每组内部是**顺序执行**的——并行来自
    // "多个任务同时在池里"，而不是单个任务内部的图内并行。两处并行叠加
    // 会向同一个池嵌套提交，实测会死锁（见 executor.hpp 的说明）。
    std::printf("\n-- 异步：16 组任务（每组 sort 256K），%u 线程 --\n", hw);
    {
        const std::size_t n = 262144;
        const int groups = 16;
        const auto data = make_data(n);

        auto g = std::make_shared<graph::Graph<float>>();
        auto g_in  = g->create_node("in", graph::NodeType::Input);
        auto g_sort= g->create_node("sort", graph::NodeType::Operator);
        auto g_out = g->create_node("out", graph::NodeType::Output);
        g->add_edge(g_in, g_sort);
        g->add_edge(g_sort, g_out);
        g->bind_op(g_sort, [](const std::vector<tensor::TensorPtr<float>>& i) {
            return ops::sort(*i[0]);
        });

        const double serial = flux_bench::best_ms(3, [&] {
            std::vector<std::unique_ptr<runtime::Executor<float>>> exs;
            for (int k = 0; k < groups; ++k) {
                auto ex = std::make_unique<runtime::Executor<float>>();
                ex->set_input(g_in, data);
                ex->run(*g);
                exs.push_back(std::move(ex));
            }
        });

        runtime::ThreadPool pool(hw);
        const double async_ms = flux_bench::best_ms(3, [&] {
            std::vector<std::unique_ptr<runtime::Executor<float>>> exs;
            std::vector<std::future<void>> futs;
            for (int k = 0; k < groups; ++k) {
                auto ex = std::make_unique<runtime::Executor<float>>();
                ex->set_input(g_in, data);
                futs.push_back(ex->run_async(*g, pool));
                exs.push_back(std::move(ex));
            }
            for (auto& f : futs) f.get();
        });

        std::printf("  串行 run()        %8.3f ms\n", serial);
        std::printf("  异步 run_async()  %8.3f ms   加速比 %5.2fx\n\n", async_ms, serial / async_ms);
    }

    // --- 池的并发分配：并行落地前的前置问题 ---
    std::printf("\n-- MemoryPool 并发分配（单次 4KB，每线程 200k 轮）--\n");
    std::printf("   （注意：这是退化场景——纯分配无计算。真实负载里分配之间有计算掩盖）\n");
    for (std::size_t threads : {1u, 2u, 4u, 8u, 16u}) {
        char name[128];
        std::snprintf(name, sizeof(name), "%2zu 线程各 200k 轮分配/释放", threads);
        flux_bench::bench(name, 5, [&] { pool_contention(threads, 200000, 1024); });
    }

    return flux_bench::report("parallel / contention");
}
