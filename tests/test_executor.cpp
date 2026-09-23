#include <flux/flux.hpp>
#include "test_util.hpp"

#include <cmath>
#include <stdexcept>
#include <future>
#include <memory>
#include <type_traits>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

using namespace flux;

// Executor 不持有线程池等独占资源，因此可自由拷贝/移动。
// 这同时是一道防线：若有人把 ThreadPool 按值塞回 Executor，ThreadPool 持有
// vector<thread> 且声明了析构函数，会让 Executor 丧失移动能力，这里立刻编译失败。
static_assert(std::is_move_constructible<runtime::Executor<float>>::value,
              "Executor 应当可移动");
static_assert(std::is_copy_constructible<runtime::Executor<float>>::value,
              "Executor 应当可拷贝");

namespace {

// 图里的算子节点数。用于把"一次 run 该物化几个张量缓冲区"写成不变量，
// 而不是硬编码数字。
std::size_t count_operators(const graph::Graph<float>& g) {
    std::size_t n = 0;
    for (const auto& node : g.nodes()) {
        if (node->type == graph::NodeType::Operator) ++n;
    }
    return n;
}

// 逐元素比较，NaN 视为相等（shift 会产出 NaN，用 == 直接比会误报）
bool same(const tensor::Tensor<float>& a, const tensor::Tensor<float>& b) {
    if (a.numel() != b.numel()) return false;
    for (std::size_t i = 0; i < a.numel(); ++i) {
        const float x = a[i], y = b[i];
        if (std::isnan(x) && std::isnan(y)) continue;
        if (x != y) return false;
    }
    return true;
}

void test_factor_graph() {
    // close -> shift(1) -> sub(close - shift) -> rolling_mean(3) -> factor
    graph::Graph<float> g;
    auto n_input  = g.create_node("Input(Close)", graph::NodeType::Input);
    auto n_shift  = g.create_node("Operator(Shift)", graph::NodeType::Operator);
    auto n_sub    = g.create_node("Operator(Sub)", graph::NodeType::Operator);
    auto n_roll   = g.create_node("Operator(RollingMean)", graph::NodeType::Operator);
    auto n_output = g.create_node("Output(Factor)", graph::NodeType::Output);

    g.add_edge(n_input, n_shift);
    g.add_edge(n_input, n_sub);   // 先连 input：sub 的 ins[0] = close
    g.add_edge(n_shift, n_sub);
    g.add_edge(n_sub, n_roll);
    g.add_edge(n_roll, n_output);

    g.bind_op(n_shift, [](const std::vector<tensor::TensorPtr<float>>& ins) {
        if (ins.size() != 1) throw std::runtime_error("shift expects 1 input");
        return ops::shift(*ins[0], 1);
    });
    g.bind_op(n_sub, [](const std::vector<tensor::TensorPtr<float>>& ins) {
        if (ins.size() != 2) throw std::runtime_error("sub expects 2 inputs");
        return ops::sub(*ins[0], *ins[1]);  // close - shift(close)
    });
    g.bind_op(n_roll, [](const std::vector<tensor::TensorPtr<float>>& ins) {
        if (ins.size() != 1) throw std::runtime_error("rolling_mean expects 1 input");
        return ops::rolling_mean(*ins[0], 3);
    });

    runtime::Executor<float> executor;
    executor.set_input(n_input,
                       tensor::Tensor<float>(tensor::Shape({10}), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    executor.run(g);

    // 期望 [NaN, NaN, NaN, 1, 1, 1, 1, 1, 1, 1]
    const auto factor = executor.get_output(n_output);
    CHECK(factor->shape() == tensor::Shape({10}));
    for (std::size_t i = 0; i < 3; ++i) CHECK(std::isnan((*factor)[i]));
    for (std::size_t i = 3; i < 10; ++i) CHECK_NEAR((*factor)[i], 1.f, 1e-6f);

    // 中间节点输出也可以取到
    const auto sub_out = executor.get_output(n_sub);
    CHECK(std::isnan((*sub_out)[0]));
    CHECK_NEAR((*sub_out)[3], 1.f, 1e-6f);
}

void test_multi_input_add() {
    graph::Graph<float> g;
    auto a   = g.create_node("Input A", graph::NodeType::Input);
    auto b   = g.create_node("Input B", graph::NodeType::Input);
    auto add = g.create_node("Operator(Add)", graph::NodeType::Operator);
    auto out = g.create_node("Output", graph::NodeType::Output);

    g.add_edge(a, add);
    g.add_edge(b, add);
    g.add_edge(add, out);

    g.bind_op(add, [](const std::vector<tensor::TensorPtr<float>>& ins) {
        if (ins.size() != 2) throw std::runtime_error("add expects 2 inputs");
        return ops::add(*ins[0], *ins[1]);
    });

    runtime::Executor<float> executor;
    executor.set_input(a, tensor::Tensor<float>(tensor::Shape({3}), {1, 2, 3}));
    executor.set_input(b, tensor::Tensor<float>(tensor::Shape({3}), {10, 20, 30}));
    executor.run(g);

    const auto r = executor.get_output(out);
    CHECK_EQ((*r)[0], 11.f);
    CHECK_EQ((*r)[1], 22.f);
    CHECK_EQ((*r)[2], 33.f);
}

void test_fanout_shares_input() {
    // 扇出：一个输入喂两个消费者。两个下游共享同一份上游张量，
    // 各自的算子产出独立的新张量，互不影响。
    graph::Graph<float> g;
    auto x      = g.create_node("Input X", graph::NodeType::Input);
    auto sh     = g.create_node("Operator(Shift)", graph::NodeType::Operator);
    auto ml     = g.create_node("Operator(Mul)", graph::NodeType::Operator);
    auto out_sh = g.create_node("Output Shift", graph::NodeType::Output);
    auto out_ml = g.create_node("Output Mul", graph::NodeType::Output);

    g.add_edge(x, sh);
    g.add_edge(x, ml);
    g.add_edge(sh, out_sh);
    g.add_edge(ml, out_ml);

    g.bind_op(sh, [](const std::vector<tensor::TensorPtr<float>>& ins) {
        if (ins.size() != 1) throw std::runtime_error("shift expects 1 input");
        return ops::shift(*ins[0], 1);
    });
    g.bind_op(ml, [](const std::vector<tensor::TensorPtr<float>>& ins) {
        if (ins.size() != 1) throw std::runtime_error("mul expects 1 input");
        return ops::mul(*ins[0], *ins[0]);
    });

    tensor::Tensor<float> xv(tensor::Shape({4}), {1, 2, 3, 4});

    runtime::Executor<float> executor;
    executor.set_input(x, xv);
    executor.run(g);

    const auto shifted = executor.get_output(out_sh);
    CHECK(std::isnan((*shifted)[0]));
    CHECK_EQ((*shifted)[1], 1.f);
    CHECK_EQ((*shifted)[2], 2.f);
    CHECK_EQ((*shifted)[3], 3.f);

    const auto squared = executor.get_output(out_ml);
    CHECK_EQ((*squared)[0], 1.f);
    CHECK_EQ((*squared)[1], 4.f);
    CHECK_EQ((*squared)[2], 9.f);
    CHECK_EQ((*squared)[3], 16.f);

    // 输入张量未被移动/破坏。注意：这条现在由 set_input 的拷贝保证，
    // 边上的共享不再可能触及调用方的张量（共享的是图自己那份 const 副本）。
    CHECK_EQ(xv[0], 1.f);
    CHECK_EQ(xv[3], 4.f);
}

void test_input_snapshot_isolated_from_caller() {
    // set_input 必须拷贝：图看到的是注入那一刻的快照。
    // 调用方之后写自己的张量，不得影响已注入的数据——否则不可变不变量
    // 就会经由"调用方仍持有可变句柄"这条路被击穿。
    graph::Graph<float> g;
    auto in  = g.create_node("Input", graph::NodeType::Input);
    auto out = g.create_node("Output", graph::NodeType::Output);
    g.add_edge(in, out);

    tensor::Tensor<float> xv(tensor::Shape({3}), {1, 2, 3});

    runtime::Executor<float> executor;
    executor.set_input(in, xv);

    xv[0] = 999.f;   // 注入之后再改源张量

    executor.run(g);
    const auto r = executor.get_output(out);
    CHECK_EQ((*r)[0], 1.f);   // 图必须看到注入时的旧值
    CHECK_EQ((*r)[2], 3.f);
}

void test_int_shift_explicit_fill() {
    // 整数 T 的 quiet_NaN() 不是真 NaN，必须显式 fill_value
    graph::Graph<int> g;
    auto in  = g.create_node("Input", graph::NodeType::Input);
    auto sh  = g.create_node("Operator(Shift)", graph::NodeType::Operator);
    auto out = g.create_node("Output", graph::NodeType::Output);

    g.add_edge(in, sh);
    g.add_edge(sh, out);

    g.bind_op(sh, [](const std::vector<tensor::TensorPtr<int>>& ins) {
        if (ins.size() != 1) throw std::runtime_error("shift expects 1 input");
        return ops::shift(*ins[0], 1, 0, 0);
    });

    runtime::Executor<int> executor;
    executor.set_input(in, tensor::Tensor<int>(tensor::Shape({3}), {1, 2, 3}));
    executor.run(g);

    const auto r = executor.get_output(out);
    CHECK_EQ((*r)[0], 0);
    CHECK_EQ((*r)[1], 1);
    CHECK_EQ((*r)[2], 2);
}

void test_missing_input_throws() {
    graph::Graph<float> g;
    auto in  = g.create_node("Input", graph::NodeType::Input);
    auto out = g.create_node("Output", graph::NodeType::Output);
    g.add_edge(in, out);

    runtime::Executor<float> executor;
    bool threw = false;
    try {
        executor.run(g);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_missing_op_throws() {
    graph::Graph<float> g;
    auto in  = g.create_node("Input", graph::NodeType::Input);
    auto op  = g.create_node("Operator", graph::NodeType::Operator);
    auto out = g.create_node("Output", graph::NodeType::Output);
    g.add_edge(in, op);
    g.add_edge(op, out);

    runtime::Executor<float> executor;
    executor.set_input(in, tensor::Tensor<float>(tensor::Shape({2}), {1, 2}));
    bool threw = false;
    try {
        executor.run(g);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_set_input_wrong_type_throws() {
    graph::Graph<float> g;
    auto op = g.create_node("Operator", graph::NodeType::Operator);

    runtime::Executor<float> executor;
    bool threw = false;
    try {
        executor.set_input(op, tensor::Tensor<float>(tensor::Shape({1}), {1.f}));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_output_wrong_arity_throws() {
    graph::Graph<float> g;
    auto out = g.create_node("Output", graph::NodeType::Output);  // 0 个输入

    runtime::Executor<float> executor;
    bool threw = false;
    try {
        executor.run(g);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_cycle_throws() {
    graph::Graph<float> g;
    auto in = g.create_node("Input", graph::NodeType::Input);
    auto a  = g.create_node("Operator A", graph::NodeType::Operator);
    auto b  = g.create_node("Operator B", graph::NodeType::Operator);
    g.add_edge(in, a);
    g.add_edge(a, b);
    g.add_edge(b, a);  // 回边构成环

    g.bind_op(a, [](const std::vector<tensor::TensorPtr<float>>&) { return tensor::Tensor<float>(tensor::Shape({2})); });
    g.bind_op(b, [](const std::vector<tensor::TensorPtr<float>>&) { return tensor::Tensor<float>(tensor::Shape({2})); });

    runtime::Executor<float> executor;
    executor.set_input(in, tensor::Tensor<float>(tensor::Shape({2}), {1, 2}));
    bool threw = false;
    try {
        executor.run(g);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);

    // 主动打断所有权环（inputs 为强引用，a↔b 互持会在析构时泄漏）
    a->inputs.clear();
    b->inputs.clear();
}

void test_get_output_before_run_throws() {
    graph::Graph<float> g;
    auto out = g.create_node("Output", graph::NodeType::Output);

    runtime::Executor<float> executor;
    bool threw = false;
    try {
        (void)executor.get_output(out);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_rerun_new_inputs() {
    graph::Graph<float> g;
    auto in  = g.create_node("Input", graph::NodeType::Input);
    auto id  = g.create_node("Operator(Identity)", graph::NodeType::Operator);
    auto out = g.create_node("Output", graph::NodeType::Output);
    g.add_edge(in, id);
    g.add_edge(id, out);

    g.bind_op(id, [](const std::vector<tensor::TensorPtr<float>>& ins) {
        if (ins.size() != 1) throw std::runtime_error("identity expects 1 input");
        // 有意拷贝：透传型算子仍会物化一个新张量（OpFn 的返回是按值的）。
        return *ins[0];
    });

    runtime::Executor<float> executor;
    executor.set_input(in, tensor::Tensor<float>(tensor::Shape({3}), {1, 2, 3}));
    executor.run(g);
    const auto r1 = executor.get_output(out);   // 按值持有所有权
    CHECK_EQ((*r1)[0], 1.f);
    CHECK_EQ((*r1)[2], 3.f);

    executor.set_input(in, tensor::Tensor<float>(tensor::Shape({3}), {4, 5, 6}));
    executor.run(g);
    const auto r2 = executor.get_output(out);
    CHECK_EQ((*r2)[0], 4.f);
    CHECK_EQ((*r2)[2], 6.f);

    // 回归：r1 跨第二次 run() 依然有效——shared_ptr 自己持有所有权。
    // 这两条断言在旧的"get_output 返回引用"接口下是 use-after-free
    // （run() 里 values_.clear() 会让引用悬垂），ASan 下必炸。
    CHECK_EQ((*r1)[0], 1.f);
    CHECK_EQ((*r1)[2], 3.f);
}

void test_output_passthrough_shares() {
    // Output 节点是零成本透传：它与上游返回的是同一个张量对象，不是副本。
    graph::Graph<float> g;
    auto in  = g.create_node("Input", graph::NodeType::Input);
    auto out = g.create_node("Output", graph::NodeType::Output);
    g.add_edge(in, out);

    runtime::Executor<float> executor;
    executor.set_input(in, tensor::Tensor<float>(tensor::Shape({3}), {1, 2, 3}));
    executor.run(g);

    CHECK(executor.get_output(out).get() == executor.get_output(in).get());
}

void test_no_edge_copies() {
    // 零拷贝的核心验收：一次 run() 物化出的张量缓冲区个数 == 算子节点数。
    // Input 与 Output 节点只共享句柄，不产生任何新缓冲区。
    //
    // 这个断言不依赖 benchmark，且期望值从图推导而非硬编码——
    // 任何把边拷贝重新引入的改动都会立刻让它失败。
    graph::Graph<float> g;
    auto x      = g.create_node("Input X", graph::NodeType::Input);
    auto sh     = g.create_node("Operator(Shift)", graph::NodeType::Operator);
    auto ml     = g.create_node("Operator(Mul)", graph::NodeType::Operator);
    auto out_sh = g.create_node("Output Shift", graph::NodeType::Output);
    auto out_ml = g.create_node("Output Mul", graph::NodeType::Output);

    g.add_edge(x, sh);
    g.add_edge(x, ml);
    g.add_edge(sh, out_sh);
    g.add_edge(ml, out_ml);

    g.bind_op(sh, [](const std::vector<tensor::TensorPtr<float>>& ins) {
        if (ins.size() != 1) throw std::runtime_error("shift expects 1 input");
        return ops::shift(*ins[0], 1);
    });
    g.bind_op(ml, [](const std::vector<tensor::TensorPtr<float>>& ins) {
        if (ins.size() != 1) throw std::runtime_error("mul expects 1 input");
        return ops::mul(*ins[0], *ins[0]);
    });

    const std::size_t expected = count_operators(g);

    runtime::Executor<float> executor;
    executor.set_input(x, tensor::Tensor<float>(tensor::Shape({4}), {1, 2, 3, 4}));

    // 快照紧贴 run()：计数器是全局的，窗口内不得构造任何张量。
    const auto before = memory::AllocationCounter::snapshot();
    executor.run(g);
    const auto after = memory::AllocationCounter::snapshot();

    CHECK_EQ(after.calls - before.calls, expected);
}

void test_executor_is_container_friendly() {
    // 回归：Executor 曾经因为按值持有 ThreadPool 而既不可拷贝也不可移动，
    // 下面这个 vector + reserve 的写法当时根本编译不过。
    // 顺带也说明构造 Executor 不再会拉起满核线程（实测曾是 1 -> 17）。
    std::vector<runtime::Executor<float>> executors;
    executors.reserve(4);
    for (int i = 0; i < 4; ++i) executors.emplace_back();

    graph::Graph<float> g;
    auto in  = g.create_node("Input", graph::NodeType::Input);
    auto out = g.create_node("Output", graph::NodeType::Output);
    g.add_edge(in, out);

    executors[0].set_input(in, tensor::Tensor<float>(tensor::Shape({2}), {7, 8}));
    executors[0].run(g);
    CHECK_EQ((*executors[0].get_output(out))[0], 7.f);
    CHECK_EQ((*executors[0].get_output(out))[1], 8.f);

    // 拷一份独立执行，互不干扰
    runtime::Executor<float> copy = executors[0];
    copy.run(g);
    CHECK_EQ((*copy.get_output(out))[0], 7.f);
}

void test_set_input_materializes_exactly_one() {
    // 注入是"每次一个缓冲区"，与图的规模无关：这条把 set_input 的拷贝
    // 钉成契约（见 test_input_snapshot_isolated_from_caller 说明它为何必要）。
    graph::Graph<float> g;
    auto in  = g.create_node("Input", graph::NodeType::Input);
    auto out = g.create_node("Output", graph::NodeType::Output);
    g.add_edge(in, out);

    runtime::Executor<float> executor;

    const auto before = memory::AllocationCounter::snapshot();
    executor.set_input(in, tensor::Tensor<float>(tensor::Shape({4}), {1, 2, 3, 4}));
    const auto after = memory::AllocationCounter::snapshot();

    // 传入的是临时量，它自身在窗口内也要物化一次，故总计 2：
    // 临时量 1 + set_input 内部的拷贝 1。
    CHECK_EQ(after.calls - before.calls, static_cast<std::size_t>(2));
}

// ---------------------------------------------------------------------------
// 按依赖分层 + 并行执行
// ---------------------------------------------------------------------------

void test_topological_layers() {
    // 1 输入 -> 2 个互不依赖的算子 -> 2 输出，应当分成 3 层
    graph::Graph<float> g;
    auto in  = g.create_node("in", graph::NodeType::Input);
    auto sh  = g.create_node("sh", graph::NodeType::Operator);
    auto ml  = g.create_node("ml", graph::NodeType::Operator);
    auto osh = g.create_node("osh", graph::NodeType::Output);
    auto oml = g.create_node("oml", graph::NodeType::Output);
    g.add_edge(in, sh);  g.add_edge(in, ml);
    g.add_edge(sh, osh); g.add_edge(ml, oml);

    auto layers = g.topological_layers();
    CHECK_EQ(layers.size(), static_cast<std::size_t>(3));
    CHECK_EQ(layers[0].size(), static_cast<std::size_t>(1));   // in
    CHECK_EQ(layers[1].size(), static_cast<std::size_t>(2));   // sh, ml 可并行
    CHECK_EQ(layers[2].size(), static_cast<std::size_t>(2));   // osh, oml
    CHECK_EQ(layers[0][0]->id, in->id);
}

void test_topological_layers_chain_is_linear() {
    // 链式图每层只有一个节点——没有并行空间，也没有分层错误
    graph::Graph<float> g;
    auto a = g.create_node("a", graph::NodeType::Input);
    auto b = g.create_node("b", graph::NodeType::Operator);
    auto c = g.create_node("c", graph::NodeType::Operator);
    g.add_edge(a, b); g.add_edge(b, c);

    auto layers = g.topological_layers();
    CHECK_EQ(layers.size(), static_cast<std::size_t>(3));
    for (const auto& l : layers) CHECK_EQ(l.size(), static_cast<std::size_t>(1));
}

void test_topological_layers_cycle_throws() {
    // 分层是执行调度的一部分，环必须在这里拦下（不像拓扑排序那样留到执行期）
    graph::Graph<float> g;
    auto a = g.create_node("a", graph::NodeType::Operator);
    auto b = g.create_node("b", graph::NodeType::Operator);
    g.add_edge(a, b);
    g.add_edge(b, a);

    bool threw = false;
    try {
        (void)g.topological_layers();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);

    a->inputs.clear();   // 打断所有权环，避免析构泄漏
    b->inputs.clear();
}

void test_parallel_matches_sequential() {
    // 同一张扇出图，两条路径必须给出完全相同的结果
    graph::Graph<float> g;
    auto in  = g.create_node("in", graph::NodeType::Input);
    auto sh  = g.create_node("sh", graph::NodeType::Operator);
    auto ml  = g.create_node("ml", graph::NodeType::Operator);
    auto osh = g.create_node("osh", graph::NodeType::Output);
    auto oml = g.create_node("oml", graph::NodeType::Output);
    g.add_edge(in, sh);  g.add_edge(in, ml);
    g.add_edge(sh, osh); g.add_edge(ml, oml);
    g.bind_op(sh, [](const std::vector<tensor::TensorPtr<float>>& i) { return ops::shift(*i[0], 1); });
    g.bind_op(ml, [](const std::vector<tensor::TensorPtr<float>>& i) { return ops::mul(*i[0], *i[0]); });

    // 注意 Tensor 的双参构造接受的是 ContainerType（带对齐分配器的 vector），
    // 不是裸 std::vector<float>，所以这里构造后填充。
    tensor::Tensor<float> data(tensor::Shape({64}));
    for (std::size_t i = 0; i < data.numel(); ++i) data[i] = 3.f;

    runtime::Executor<float> seq;
    seq.set_input(in, data);
    seq.run(g);
    const auto seq_sh = seq.get_output(osh);
    const auto seq_ml = seq.get_output(oml);

    runtime::Executor<float> par;
    par.set_input(in, data);
    runtime::ThreadPool pool(4);
    par.run_parallel(g, pool);
    const auto par_sh = par.get_output(osh);
    const auto par_ml = par.get_output(oml);

    CHECK(same(*seq_sh, *par_sh));
    CHECK(same(*seq_ml, *par_ml));
    CHECK_EQ((*par_ml)[0], 9.f);   // 3 * 3
}

void test_parallel_chain_gives_same_result() {
    // 链式图在并行路径下退化为逐层单节点执行，结果必须一致
    graph::Graph<float> g;
    auto in  = g.create_node("in", graph::NodeType::Input);
    auto a   = g.create_node("a", graph::NodeType::Operator);
    auto b   = g.create_node("b", graph::NodeType::Operator);
    auto out = g.create_node("out", graph::NodeType::Output);
    g.add_edge(in, a); g.add_edge(a, b); g.add_edge(b, out);
    g.bind_op(a, [](const std::vector<tensor::TensorPtr<float>>& i) { return ops::mul(*i[0], *i[0]); });
    g.bind_op(b, [](const std::vector<tensor::TensorPtr<float>>& i) { return ops::shift(*i[0], 1); });

    runtime::Executor<float> ex;
    ex.set_input(in, tensor::Tensor<float>(tensor::Shape({4}), {1, 2, 3, 4}));
    runtime::ThreadPool pool(4);
    ex.run_parallel(g, pool);

    const auto r = ex.get_output(out);
    CHECK(std::isnan((*r)[0]));
    CHECK_EQ((*r)[1], 1.f);    // 1*1
    CHECK_EQ((*r)[2], 4.f);    // 2*2
    CHECK_EQ((*r)[3], 9.f);    // 3*3
}

void test_parallel_propagates_exception() {
    // 层内某个节点抛异常，必须从 run_parallel 抛出来而不是被 worker 吞掉
    graph::Graph<float> g;
    auto in   = g.create_node("in", graph::NodeType::Input);
    auto bad  = g.create_node("bad", graph::NodeType::Operator);
    auto good = g.create_node("good", graph::NodeType::Operator);
    auto o1   = g.create_node("o1", graph::NodeType::Output);
    auto o2   = g.create_node("o2", graph::NodeType::Output);
    g.add_edge(in, bad);  g.add_edge(in, good);
    g.add_edge(bad, o1);  g.add_edge(good, o2);

    g.bind_op(bad, [](const std::vector<tensor::TensorPtr<float>>&) -> tensor::Tensor<float> {
        throw std::runtime_error("boom");
    });
    g.bind_op(good, [](const std::vector<tensor::TensorPtr<float>>& i) { return ops::mul(*i[0], *i[0]); });

    runtime::Executor<float> ex;
    ex.set_input(in, tensor::Tensor<float>(tensor::Shape({4}), {1, 2, 3, 4}));
    runtime::ThreadPool pool(4);

    bool threw = false;
    try {
        ex.run_parallel(g, pool);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_parallel_rerun_is_stable() {
    // 可重复调用，且与串行路径结果一致
    graph::Graph<float> g;
    auto in  = g.create_node("in", graph::NodeType::Input);
    auto m1  = g.create_node("m1", graph::NodeType::Operator);
    auto m2  = g.create_node("m2", graph::NodeType::Operator);
    auto out = g.create_node("out", graph::NodeType::Output);
    g.add_edge(in, m1); g.add_edge(in, m2); g.add_edge(m1, out);
    g.bind_op(m1, [](const std::vector<tensor::TensorPtr<float>>& i) { return ops::mul(*i[0], *i[0]); });
    g.bind_op(m2, [](const std::vector<tensor::TensorPtr<float>>& i) { return ops::mul(*i[0], *i[0]); });

    runtime::Executor<float> ex;
    runtime::ThreadPool pool(4);
    ex.set_input(in, tensor::Tensor<float>(tensor::Shape({4}), {1, 2, 3, 4}));

    ex.run_parallel(g, pool);
    const auto r1 = ex.get_output(out);

    ex.set_input(in, tensor::Tensor<float>(tensor::Shape({4}), {5, 6, 7, 8}));
    ex.run_parallel(g, pool);
    const auto r2 = ex.get_output(out);

    CHECK_EQ((*r1)[0], 1.f);
    CHECK_EQ((*r2)[0], 25.f);
    // r1 跨第二次 run 依然有效（get_output 返回持有所有权的句柄）
    CHECK_EQ((*r1)[0], 1.f);
}

// ---------------------------------------------------------------------------
// 异步执行
// ---------------------------------------------------------------------------

void test_async_matches_sync() {
    graph::Graph<float> g;
    auto in  = g.create_node("in", graph::NodeType::Input);
    auto m   = g.create_node("m", graph::NodeType::Operator);
    auto out = g.create_node("out", graph::NodeType::Output);
    g.add_edge(in, m); g.add_edge(m, out);
    g.bind_op(m, [](const std::vector<tensor::TensorPtr<float>>& i) { return ops::mul(*i[0], *i[0]); });

    const tensor::Tensor<float> data(tensor::Shape({4}), {1, 2, 3, 4});

    runtime::Executor<float> sync;
    sync.set_input(in, data);
    sync.run(g);
    const auto expect = sync.get_output(out);

    // 注意：Executor 必须活到 future 就绪——契约之一
    runtime::Executor<float> async;
    async.set_input(in, data);
    runtime::ThreadPool pool(2);
    auto fut = async.run_async(g, pool);
    fut.get();   // 就绪后才能取结果

    CHECK(same(*expect, *async.get_output(out)));
}

void test_async_propagates_exception() {
    // 节点里抛的异常必须经由 future 重抛，不能被后台线程吞掉
    graph::Graph<float> g;
    auto in  = g.create_node("in", graph::NodeType::Input);
    auto bad = g.create_node("bad", graph::NodeType::Operator);
    auto out = g.create_node("out", graph::NodeType::Output);
    g.add_edge(in, bad); g.add_edge(bad, out);
    g.bind_op(bad, [](const std::vector<tensor::TensorPtr<float>>&) -> tensor::Tensor<float> {
        throw std::runtime_error("boom");
    });

    runtime::Executor<float> ex;
    ex.set_input(in, tensor::Tensor<float>(tensor::Shape({2}), {1, 2}));
    runtime::ThreadPool pool(2);

    bool threw = false;
    try {
        ex.run_async(g, pool).get();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_async_multiple_tasks_concurrently() {
    // 异步的主要用法：多组任务同时在池里跑。这里用 4 组参数验证各自结果正确
    // ——并发正确性由 TSan 覆盖，这里盯的是"结果没有串台"。
    auto make_graph = [](std::shared_ptr<graph::Node<float>>& in,
                         std::shared_ptr<graph::Node<float>>& out) {
        auto g = std::make_shared<graph::Graph<float>>();
        in  = g->create_node("in", graph::NodeType::Input);
        auto m = g->create_node("m", graph::NodeType::Operator);
        out = g->create_node("out", graph::NodeType::Output);
        g->add_edge(in, m); g->add_edge(m, out);
        g->bind_op(m, [](const std::vector<tensor::TensorPtr<float>>& i) { return ops::mul(*i[0], *i[0]); });
        return g;
    };

    runtime::ThreadPool pool(4);

    // Executor 与 Graph 都要活到 future 就绪，所以先全部建好再一并 get()
    std::vector<std::shared_ptr<graph::Graph<float>>> graphs;
    std::vector<std::shared_ptr<graph::Node<float>>> ins, outs;
    std::vector<std::unique_ptr<runtime::Executor<float>>> execs;
    std::vector<std::future<void>> futures;

    const float seeds[4] = {2.f, 3.f, 4.f, 5.f};
    for (int k = 0; k < 4; ++k) {
        std::shared_ptr<graph::Node<float>> in, out;
        graphs.push_back(make_graph(in, out));
        ins.push_back(in);
        outs.push_back(out);

        auto ex = std::make_unique<runtime::Executor<float>>();
        ex->set_input(in, tensor::Tensor<float>(tensor::Shape({3}), {seeds[k], seeds[k], seeds[k]}));
        futures.push_back(ex->run_async(*graphs.back(), pool));
        execs.push_back(std::move(ex));
    }

    for (auto& f : futures) f.get();

    for (int k = 0; k < 4; ++k) {
        const auto r = execs[k]->get_output(outs[k]);
        CHECK_EQ((*r)[0], seeds[k] * seeds[k]);
    }
}

void test_thread_pool_zero_threads() {
    // 即使入参为 0 或硬件并发探测为 0，也不应产生死锁挂起，应至少有 1 个 worker 线程
    runtime::ThreadPool pool(0);
    auto fut = pool.enqueue([] { return 42; });
    CHECK_EQ(fut.get(), 42);
}

void test_null_node_guards() {
    runtime::Executor<float> ex;
    tensor::Tensor<float> t(tensor::Shape({2}), {1.f, 2.f});
    bool threw_set = false;
    try {
        ex.set_input(nullptr, t);
    } catch (const std::invalid_argument&) {
        threw_set = true;
    }
    CHECK(threw_set);

    bool threw_get = false;
    try {
        (void)ex.get_output(nullptr);
    } catch (const std::invalid_argument&) {
        threw_get = true;
    }
    CHECK(threw_get);

    graph::Graph<float> g;
    auto node = g.create_node("a", graph::NodeType::Operator);
    bool threw_edge = false;
    try {
        g.add_edge(nullptr, node);
    } catch (const std::invalid_argument&) {
        threw_edge = true;
    }
    CHECK(threw_edge);

    bool threw_bind = false;
    try {
        g.bind_op(nullptr, [](const auto&) { return tensor::Tensor<float>(tensor::Shape({1})); });
    } catch (const std::invalid_argument&) {
        threw_bind = true;
    }
    CHECK(threw_bind);
}

void test_parallel_propagates_exception_waits_for_all() {
    // 层内某节点抛出异常时，屏障必须等待层内其他耗时任务全部结束，
    // 绝不能提前展开栈帧导致 results 局部变量发生 use-after-free
    graph::Graph<float> g;
    auto in   = g.create_node("in", graph::NodeType::Input);
    auto bad  = g.create_node("bad", graph::NodeType::Operator);
    auto slow = g.create_node("slow", graph::NodeType::Operator);
    auto o1   = g.create_node("o1", graph::NodeType::Output);
    auto o2   = g.create_node("o2", graph::NodeType::Output);
    g.add_edge(in, bad);   g.add_edge(in, slow);
    g.add_edge(bad, o1);   g.add_edge(slow, o2);

    g.bind_op(bad, [](const std::vector<tensor::TensorPtr<float>>&) -> tensor::Tensor<float> {
        throw std::runtime_error("bad node error");
    });
    std::atomic<bool> slow_completed{false};
    g.bind_op(slow, [&slow_completed](const std::vector<tensor::TensorPtr<float>>& ins) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        slow_completed.store(true);
        return *ins[0];
    });

    runtime::Executor<float> ex;
    ex.set_input(in, tensor::Tensor<float>(tensor::Shape({2}), {1.f, 2.f}));
    runtime::ThreadPool pool(2);

    bool threw = false;
    try {
        ex.run_parallel(g, pool);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(slow_completed.load());
}

} // namespace

int main() {
    test_factor_graph();
    test_multi_input_add();
    test_fanout_shares_input();
    test_input_snapshot_isolated_from_caller();
    test_int_shift_explicit_fill();
    test_missing_input_throws();
    test_missing_op_throws();
    test_set_input_wrong_type_throws();
    test_output_wrong_arity_throws();
    test_cycle_throws();
    test_get_output_before_run_throws();
    test_rerun_new_inputs();
    test_output_passthrough_shares();
    test_no_edge_copies();
    test_executor_is_container_friendly();
    test_set_input_materializes_exactly_one();
    test_topological_layers();
    test_topological_layers_chain_is_linear();
    test_topological_layers_cycle_throws();
    test_parallel_matches_sequential();
    test_parallel_chain_gives_same_result();
    test_parallel_propagates_exception();
    test_parallel_propagates_exception_waits_for_all();
    test_parallel_rerun_is_stable();
    test_async_matches_sync();
    test_async_propagates_exception();
    test_async_multiple_tasks_concurrently();
    test_thread_pool_zero_threads();
    test_null_node_guards();
    return flux_test::summary();
}
