#pragma once

#include <flux/tensor/tensor.hpp>
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace flux::graph {

enum class NodeType { Input, Operator, Output };

// 计算图节点。
// op 承载"输入张量 -> 输出张量"的可执行闭包，标量参数（shift 偏移、窗口大小等）
// 在绑定阶段按值捕获进闭包；Input 节点的 op 为空，数据由 Executor 注入。
template <typename T>
struct Node {
    using OpFn = std::function<tensor::Tensor<T>(const std::vector<tensor::Tensor<T>>&)>;

    std::size_t id;
    std::string name;
    NodeType type;
    // inputs 强引用（消费者拥有生产者，保证上游存活）；outputs 弱引用——
    // 若双向都强引用，add_edge(a,b) 会让 a↔b 互持形成所有权环，节点永远无法释放。
    // 使用 outputs 时需先 lock()。
    std::vector<std::shared_ptr<Node>> inputs;   // 注入类名 == Node<T>
    std::vector<std::weak_ptr<Node>> outputs;
    OpFn op;                                     // 未绑定时为空

    Node(std::size_t id, std::string name, NodeType type)
        : id(id), name(std::move(name)), type(type) {}

    bool has_op() const { return static_cast<bool>(op); }
};

}
