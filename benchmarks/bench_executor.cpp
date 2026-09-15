// 计算图执行的开销。
//
// 核心问题：零拷贝改造（Executor 边上传 shared_ptr 而非张量值）把扇出图的
// 张量缓冲区分配次数从 7 降到了 2，但**耗时降了多少？** 分配次数减少不等于
// 时间减少——这里要把它变成毫秒。
//
// 做法：单独测一次"张量深拷贝"的成本，它就是被消除掉的那个操作。
// 于是 (7 - 2) × 单次拷贝成本 就是这次改造省下的量级。
//
// 另外用极小张量（4 个元素）跑同一张图：内核开销这时可以忽略，量到的基本
// 就是执行器自身（shared_ptr 拷贝、map 查找、std::function 调用）的成本。

#include <flux/flux.hpp>
#include "bench_util.hpp"

#include <cstdio>
#include <memory>
#include <vector>

using namespace flux;

namespace {

constexpr std::size_t kBig = 1u << 20;   // 1M float = 4 MB
constexpr std::size_t kTiny = 4;

tensor::Tensor<float> make_data(std::size_t n) {
    tensor::Tensor<float> t(tensor::Shape({n}));
    for (std::size_t i = 0; i < n; ++i) {
        // 用确定性序列而非随机数，保证结果可复现
        t[i] = static_cast<float>(i % 97) * 0.5f + 1.0f;
    }
    return t;
}

// 扇出图：1 输入 -> 2 算子 -> 2 输出
struct FanoutGraph {
    graph::Graph<float> g;
    std::shared_ptr<graph::Node<float>> in, out_shift, out_mul;

    FanoutGraph() {
        in        = g.create_node("Input", graph::NodeType::Input);
        auto sh   = g.create_node("Shift", graph::NodeType::Operator);
        auto ml   = g.create_node("Mul", graph::NodeType::Operator);
        out_shift = g.create_node("OutShift", graph::NodeType::Output);
        out_mul   = g.create_node("OutMul", graph::NodeType::Output);

        g.add_edge(in, sh);
        g.add_edge(in, ml);
        g.add_edge(sh, out_shift);
        g.add_edge(ml, out_mul);

        g.bind_op(sh, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::shift(*ins[0], 1);
        });
        g.bind_op(ml, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::mul(*ins[0], *ins[0]);
        });
    }
};

// 链式图：输入 -> add -> mul -> sub -> rolling_mean
struct ChainGraph {
    graph::Graph<float> g;
    std::shared_ptr<graph::Node<float>> in, out;

    ChainGraph() {
        in       = g.create_node("Input", graph::NodeType::Input);
        auto a   = g.create_node("Add", graph::NodeType::Operator);
        auto m   = g.create_node("Mul", graph::NodeType::Operator);
        auto s   = g.create_node("Sub", graph::NodeType::Operator);
        auto rm  = g.create_node("RollingMean", graph::NodeType::Operator);
        out      = g.create_node("Out", graph::NodeType::Output);

        g.add_edge(in, a);
        g.add_edge(a, m);
        g.add_edge(m, s);
        g.add_edge(s, rm);
        g.add_edge(rm, out);

        // 用 x + 1.0 这种自带常量的算子，避免再引入额外输入节点
        g.bind_op(a, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::add(*ins[0], *ins[0]);
        });
        g.bind_op(m, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::mul(*ins[0], *ins[0]);
        });
        g.bind_op(s, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::sub(*ins[0], *ins[0]);
        });
        g.bind_op(rm, [](const std::vector<tensor::TensorPtr<float>>& ins) {
            return ops::rolling_mean(*ins[0], 20);
        });
    }
};

} // namespace

int main() {
    const auto big = make_data(kBig);

    std::printf("数据规模：1M float = %.1f MB\n", static_cast<double>(kBig) * sizeof(float) / (1024 * 1024));

    // --- 被消除掉的那个操作本身值多少时间 ---
    flux_bench::bench("张量深拷贝 1M（零拷贝改造消除的正是它）", 30, [&] {
        tensor::Tensor<float> copy = big;
        flux_bench::keep(copy.data());
    });

    // --- 单算子基线：把内核成本与执行器成本分开 ---
    flux_bench::bench("单算子 mul 1M（内核基线，不经执行器）", 30, [&] {
        auto r = ops::mul(big, big);
        flux_bench::keep(r.data());
    });

    // --- 扇出图 ---
    {
        FanoutGraph fg;
        runtime::Executor<float> ex;
        ex.set_input(fg.in, big);

        flux_bench::bench("扇出图 1 输入 -> 2 算子 -> 2 输出（1M）", 30, [&] {
            ex.run(fg.g);
            flux_bench::keep(ex.get_output(fg.out_shift).get());
            flux_bench::keep(ex.get_output(fg.out_mul).get());
        });
    }

    // --- 链式图 ---
    {
        ChainGraph cg;
        runtime::Executor<float> ex;
        ex.set_input(cg.in, big);

        flux_bench::bench("链式图 4 算子（1M）", 30, [&] {
            ex.run(cg.g);
            flux_bench::keep(ex.get_output(cg.out).get());
        });
    }

    // --- 纯执行器开销 ---
    {
        FanoutGraph fg;
        runtime::Executor<float> ex;
        ex.set_input(fg.in, make_data(kTiny));

        flux_bench::bench("同上的扇出图，但只有 4 个元素（≈纯执行器开销）", 200, [&] {
            ex.run(fg.g);
            flux_bench::keep(ex.get_output(fg.out_shift).get());
            flux_bench::keep(ex.get_output(fg.out_mul).get());
        });
    }

    return flux_bench::report("executor");
}
