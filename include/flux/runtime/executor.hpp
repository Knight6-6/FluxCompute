#pragma once

#include <flux/graph/graph.hpp>
#include <flux/runtime/thread_pool.hpp>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <memory>

namespace flux::runtime {

// 顺序执行计算图的 CPU Executor。
// 数据流：每个节点产出张量，按 node id 存在 values_ 中；下游节点从 inputs 收集。
//
// 张量沿边以 shared_ptr<const Tensor<T>> 传递，绝不深拷贝：一个节点的输出
// 在整次 run 中只物化一次，扇出 N 的下游共享同一份数据。安全性来自一条
// 不变量——张量一旦发布即不可变（见 tensor::TensorPtr 的 const），
// 因此共享不会退化成别名 bug。
//
// 本阶段同步、顺序执行；ThreadPool 成员为后续并行调度预留。
template <typename T>
class Executor {
private:
    ThreadPool thread_pool_;                                       // 本阶段未使用，为并行调度预留
    std::unordered_map<std::size_t, tensor::TensorPtr<T>> inputs_;  // set_input 绑定，跨 run 保留
    std::unordered_map<std::size_t, tensor::TensorPtr<T>> values_;  // 本次 run 的节点输出

    // 这两个 map 永远不要用 operator[] 读。TensorPtr 可默认构造，而
    // operator[] 在未命中时会静默插入一个 null 指针，随后在算子闭包的
    // *ins[0] 处空解引用崩溃。一律用 find()，未命中就抛异常。

    // 发布：把张量交出去成为共享句柄，此后不可变。
    // 移动版用于算子新算出的张量——Tensor 的移动构造是 noexcept 且真的
    // 窃取 vector 的缓冲区指针，故零深拷贝。
    static tensor::TensorPtr<T> publish(tensor::Tensor<T>&& t) {
        return std::make_shared<const tensor::Tensor<T>>(std::move(t));
    }

    // 拷贝版用于 set_input：所有权在调用方，且图持有的副本必须独立于
    // 调用方后续对该张量的写入（否则不可变不变量当场破裂）。
    static tensor::TensorPtr<T> publish(const tensor::Tensor<T>& t) {
        return std::make_shared<const tensor::Tensor<T>>(t);
    }

    // 取已就绪的上游输出。按值返回 shared_ptr 而非引用：调用方
    // values_.emplace(...) 可能触发 rehash，指向容器内部元素的引用会悬垂。
    tensor::TensorPtr<T> lookup(const std::shared_ptr<graph::Node<T>>& node) const {
        auto it = values_.find(node->id);
        if (it == values_.end()) {
            throw std::runtime_error("executor: output of '" + node->name
                                     + "' not available (cycle or ordering bug)");
        }
        if (!it->second) {
            throw std::runtime_error("executor: null tensor for node '" + node->name + "'");
        }
        return it->second;
    }

public:
    Executor() = default;

    // 运行前为 Input 节点注入数据。重复设置时最后一次生效。
    void set_input(const std::shared_ptr<graph::Node<T>>& node, const tensor::Tensor<T>& value) {
        if (node->type != graph::NodeType::Input) {
            throw std::runtime_error("set_input: node '" + node->name + "' is not an Input");
        }
        // 拷贝一次；绝不 move，避免破坏调用方的张量。
        // 这次拷贝承担所有权与不可变性两个职责：所有权在调用方手里，
        // 而共享要求图中那份必须与调用方之后的写入无关。
        // 用 insert_or_assign 而非 operator[]：后者要先默认构造，且语义是
        // "仅插入"，与"重复设置时最后一次生效"不符。
        inputs_.insert_or_assign(node->id, publish(value));
    }

    // 顺序执行整个计算图。
    // 校验：Input 节点必须有数据；Operator 节点必须绑定了 op；Output 节点恰好一个输入；
    // 上游输出必须可用（否则为环或排序 bug）。
    // 可重复调用：inputs_ 保留，values_ 每次重建。
    void run(const graph::Graph<T>& graph) {
        values_.clear();
        // 输入共享 inputs_ 里的句柄即可：张量不可变，扇出天然安全。
        for (const auto& kv : inputs_) {
            values_.emplace(kv.first, kv.second);
        }

        auto execution_order = graph.topological_sort();
        for (const auto& node : execution_order) {
            if (node->type == graph::NodeType::Input) {
                if (values_.find(node->id) == values_.end()) {
                    throw std::runtime_error("executor: missing input data for node '" + node->name + "'");
                }
                continue;  // 数据已就位
            }

            // Output 节点透传其唯一输入：直接共享上游句柄，零拷贝、零额外分配。
            // 必须先经 lookup() 按值取出——若把 values_ 内部元素直接喂给
            // emplace，rehash 会让那个引用在构造过程中悬垂。
            if (node->type == graph::NodeType::Output) {
                if (node->inputs.size() != 1) {
                    throw std::runtime_error("executor: output node '" + node->name
                                             + "' must have exactly one input");
                }
                values_.emplace(node->id, lookup(node->inputs[0]));
                continue;
            }

            // Operator：收集上游输出。这里只拷贝 shared_ptr，不拷贝张量数据。
            std::vector<tensor::TensorPtr<T>> ins;
            ins.reserve(node->inputs.size());
            for (const auto& in : node->inputs) {
                ins.push_back(lookup(in));
            }

            if (!node->has_op()) {
                throw std::runtime_error("executor: no operator bound to node '" + node->name + "'");
            }
            values_.emplace(node->id, publish(node->op(ins)));
        }
    }

    // 取节点输出。返回的 shared_ptr 持有所有权：只要它还在，张量就一直有效，
    // 跨 run() 也成立——第二次 run 只是让 executor 不再持有它，不影响调用方这一份。
    // 张量发布后不可变，要改请先拷贝。
    tensor::TensorPtr<T> get_output(const std::shared_ptr<graph::Node<T>>& node) const {
        auto it = values_.find(node->id);
        if (it == values_.end()) {
            throw std::runtime_error("executor: no output for node '" + node->name
                                     + "' (run() not called or node unreachable)");
        }
        return it->second;
    }
};

}
