#include <flux/flux.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <stdexcept>

int main() {
    using namespace flux;

    // 1. 验证 MemoryPool 分配与回收
    void* raw_ptr = memory::MemoryPool::instance().allocate(1024);
    std::cout << "[MemoryPool] Successfully allocated 1024 bytes (64-byte aligned).\n";
    memory::MemoryPool::instance().deallocate(raw_ptr, 1024);

    // 2. 构建一个简单的 Alpha 因子计算图并真实执行
    // Raw Close -> Shift(1) -> Sub -> RollingMean(3) -> Factor
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

    // 绑定算子闭包：标量参数在构建期按值捕获
    // 闭包收到的是共享句柄（上游输出只物化一次，扇出共享），故解引用后
    // 交给算子。ops::* 本身仍是值语义，引用不会渗进算子层。
    g.bind_op(n_shift, [](const std::vector<tensor::TensorPtr<float>>& ins) {
        if (ins.size() != 1) throw std::runtime_error("shift expects 1 input");
        return ops::shift(*ins[0], /*offset=*/1);
    });
    g.bind_op(n_sub, [](const std::vector<tensor::TensorPtr<float>>& ins) {
        if (ins.size() != 2) throw std::runtime_error("sub expects 2 inputs");
        return ops::sub(*ins[0], *ins[1]);  // close - shift(close)
    });
    g.bind_op(n_roll, [](const std::vector<tensor::TensorPtr<float>>& ins) {
        if (ins.size() != 1) throw std::runtime_error("rolling_mean expects 1 input");
        return ops::rolling_mean(*ins[0], /*window=*/3);
    });

    // 3. Runtime 调度执行
    runtime::Executor<float> executor;
    tensor::Tensor<float> close(tensor::Shape({10}), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    executor.set_input(n_input, close);
    executor.run(g);

    // get_output 按值返回 shared_ptr：持有它即持有所有权，跨 run 也有效。
    const auto factor = executor.get_output(n_output);
    std::cout << "\n[FluxCompute] Factor = rolling_mean(close - shift(close, 1), window=3):\n";
    for (std::size_t i = 0; i < factor->numel(); ++i) {
        std::cout << (std::isnan((*factor)[i]) ? " nan" : (" " + std::to_string((*factor)[i])));
    }
    std::cout << std::endl;

    std::cout << "\nFluxCompute Architecture Verified Successfully!\n";
    return 0;
}
