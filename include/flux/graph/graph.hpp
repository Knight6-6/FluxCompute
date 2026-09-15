#pragma once

#include <flux/graph/node.hpp>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <utility>
#include <stdexcept>
#include <algorithm>

namespace flux::graph {

template <typename T>
class Graph {
private:
    std::vector<std::shared_ptr<Node<T>>> nodes_;
    std::size_t next_id_ = 0;

public:
    std::shared_ptr<Node<T>> create_node(const std::string& name, NodeType type) {
        auto node = std::make_shared<Node<T>>(next_id_++, name, type);
        nodes_.push_back(node);
        return node;
    }

    void add_edge(const std::shared_ptr<Node<T>>& src, const std::shared_ptr<Node<T>>& dst) {
        src->outputs.push_back(dst);
        dst->inputs.push_back(src);
    }

    // 给 Operator 节点绑定可执行闭包。绑定时机在构建期，便于尽早发现漏绑。
    void bind_op(const std::shared_ptr<Node<T>>& node, typename Node<T>::OpFn fn) {
        if (node->type != NodeType::Operator) {
            throw std::runtime_error("bind_op: node '" + node->name + "' is not an Operator");
        }
        node->op = std::move(fn);
    }

    // 拓扑排序：推导最佳执行顺序（生产者在前）。visited 前置标记保证每节点恰好出现一次。
    std::vector<std::shared_ptr<Node<T>>> topological_sort() const {
        std::vector<std::shared_ptr<Node<T>>> order;
        std::unordered_set<std::size_t> visited;

        auto dfs = [&](auto& self, const std::shared_ptr<Node<T>>& node) -> void {
            visited.insert(node->id);
            for (const auto& in : node->inputs) {
                if (visited.find(in->id) == visited.end()) {
                    self(self, in);
                }
            }
            order.push_back(node);
        };

        for (const auto& node : nodes_) {
            if (visited.find(node->id) == visited.end()) {
                dfs(dfs, node);
            }
        }
        return order;
    }

    // 按依赖分层：同一层内的节点互不依赖，可以并行执行。
    //
    // 用 Kahn 算法而不是复用上面的 DFS 后序：DFS 给出的是**全序**，
    // 分层需要按"何时就绪"分组，两者不是一回事。
    //
    // 这是 Node::outputs 的第一个消费者——它在此之前一直是只写不读的。
    // outputs 是弱引用，取用前需要 lock()。
    //
    // 有环时抛 runtime_error：Kahn 会把环上的节点永远留在非零入度里，
    // 据此判定。这与拓扑排序对环"不检测、留给执行期暴露"的做法不同——
    // 分层本身就是执行调度的一部分，环在这里必须拦下。
    std::vector<std::vector<std::shared_ptr<Node<T>>>> topological_layers() const {
        std::unordered_map<std::size_t, std::size_t> indegree;
        for (const auto& node : nodes_) {
            indegree[node->id] = node->inputs.size();
        }

        std::vector<std::shared_ptr<Node<T>>> ready;
        for (const auto& node : nodes_) {
            if (indegree[node->id] == 0) ready.push_back(node);
        }

        std::vector<std::vector<std::shared_ptr<Node<T>>>> layers;
        std::size_t placed = 0;

        while (!ready.empty()) {
            layers.push_back(ready);
            placed += ready.size();

            std::vector<std::shared_ptr<Node<T>>> next;
            for (const auto& node : ready) {
                for (const auto& weak : node->outputs) {
                    const auto dst = weak.lock();
                    if (!dst) continue;   // 下游已被销毁（弱引用允许失效）
                    if (--indegree[dst->id] == 0) next.push_back(dst);
                }
            }
            ready = std::move(next);
        }

        if (placed != nodes_.size()) {
            throw std::runtime_error("topological_layers: graph has a cycle");
        }
        return layers;
    }

    const std::vector<std::shared_ptr<Node<T>>>& nodes() const { return nodes_; }
};

}
