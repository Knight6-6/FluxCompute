#include <flux/flux.hpp>
#include "test_util.hpp"

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

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
    return flux_test::summary();
}
