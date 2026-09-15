#pragma once

#include <flux/graph/node.hpp>
#include <vector>
#include <memory>
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

    const std::vector<std::shared_ptr<Node<T>>>& nodes() const { return nodes_; }
};

}
