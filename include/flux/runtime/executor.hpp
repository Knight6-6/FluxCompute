#pragma once

#include <flux/graph/graph.hpp>
#include <flux/runtime/thread_pool.hpp>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace flux::runtime {

// 顺序执行计算图的 CPU Executor。
// 数据流：每个节点产生输出张量，按 node id 存在 values_ 中；下游节点从 inputs 收集。
// 本阶段同步、顺序执行；ThreadPool 成员为后续并行调度预留。
template <typename T>
class Executor {
private:
    ThreadPool thread_pool_;                                     // 本阶段未使用，为并行调度预留
    std::unordered_map<std::size_t, tensor::Tensor<T>> inputs_;  // set_input 绑定，跨 run 保留
    std::unordered_map<std::size_t, tensor::Tensor<T>> values_;  // 本次 run 的节点输出

public:
    Executor() = default;

    // 运行前为 Input 节点注入数据。重复设置时最后一次生效。
    void set_input(const std::shared_ptr<graph::Node<T>>& node, const tensor::Tensor<T>& value) {
        if (node->type != graph::NodeType::Input) {
            throw std::runtime_error("set_input: node '" + node->name + "' is not an Input");
        }
        // 拷贝；绝不 move，避免破坏调用方的张量。
        // 用 insert_or_assign 而非 operator[]：后者要求 Tensor 可默认构造。
        inputs_.insert_or_assign(node->id, value);
    }

    // 顺序执行整个计算图。
    // 校验：Input 节点必须有数据；Operator 节点必须绑定了 op；Output 节点恰好一个输入；
    // 上游输出必须可用（否则为环或排序 bug）。
    // 可重复调用：inputs_ 保留，values_ 每次重建。
    void run(const graph::Graph<T>& graph) {
        values_.clear();
        for (const auto& kv : inputs_) {
            values_.emplace(kv.first, kv.second);  // 拷贝输入，保证扇出安全
        }

        auto execution_order = graph.topological_sort();
        for (const auto& node : execution_order) {
            if (node->type == graph::NodeType::Input) {
                if (values_.find(node->id) == values_.end()) {
                    throw std::runtime_error("executor: missing input data for node '" + node->name + "'");
                }
                continue;  // 数据已就位
            }

            // 收集上游输出（拷贝，绝不 move：一个节点可能被多个下游消费）
            std::vector<tensor::Tensor<T>> ins;
            ins.reserve(node->inputs.size());
            for (const auto& in : node->inputs) {
                auto it = values_.find(in->id);
                if (it == values_.end()) {
                    throw std::runtime_error("executor: output of '" + in->name
                                             + "' not available (cycle or ordering bug)");
                }
                ins.push_back(it->second);
            }

            if (node->type == graph::NodeType::Output) {
                // Output 节点透传其唯一输入
                if (node->inputs.size() != 1) {
                    throw std::runtime_error("executor: output node '" + node->name
                                             + "' must have exactly one input");
                }
                values_.emplace(node->id, std::move(ins[0]));  // ins 是本地临时量，可移动
                continue;
            }

            // Operator
            if (!node->has_op()) {
                throw std::runtime_error("executor: no operator bound to node '" + node->name + "'");
            }
            values_.emplace(node->id, node->op(ins));
        }
    }

    // 取节点输出。返回的引用在下次 run() 之前有效；需要跨 run 保存请自行拷贝。
    const tensor::Tensor<T>& get_output(const std::shared_ptr<graph::Node<T>>& node) const {
        auto it = values_.find(node->id);
        if (it == values_.end()) {
            throw std::runtime_error("executor: no output for node '" + node->name
                                     + "' (run() not called or node unreachable)");
        }
        return it->second;
    }
};

}
