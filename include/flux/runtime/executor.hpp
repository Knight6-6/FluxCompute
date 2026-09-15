#pragma once

#include <flux/graph/graph.hpp>
#include <flux/runtime/thread_pool.hpp>
#include <future>
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
// 本阶段同步、顺序执行，Executor 不持有任何独占资源——它只是两个 map，
// 因此可以廉价创建、自由拷贝/移动。
//
// 关于线程池：这里刻意**不**持有 ThreadPool。按值内嵌一个线程池意味着每构造
// 一个 Executor 就起满 hardware_concurrency 个线程，而当前一个都不用（实测：
// 构造 1 个 Executor 让进程线程数从 1 涨到 17，12 个则到 193），同时因为
// ThreadPool 持有 vector<thread> 且声明了析构函数，Executor 会连带变成
// 不可拷贝、不可移动，连放进 std::vector 都编不过。
//
// 并行调度落地时（按依赖层并行）再引入线程池，届时才谈得上决定它的归属：
// 每个 Executor 一个、外部注入、还是进程级共享。在没有消费者之前先定这个
// 是空想，所以先不建机制。
template <typename T>
class Executor {
private:
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

    // 执行单个节点，返回它的输出。顺序执行与并行执行共用这一份逻辑，
    // 保证两条路径的校验、错误信息、语义完全一致。
    //
    // 只读 values_（经 lookup），不写——并行版由调用方在层结束后统一合并，
    // 见 run_parallel 的说明。
    tensor::TensorPtr<T> execute_node(const std::shared_ptr<graph::Node<T>>& node) const {
        if (node->type == graph::NodeType::Input) {
            auto it = values_.find(node->id);
            if (it == values_.end()) {
                throw std::runtime_error("executor: missing input data for node '" + node->name + "'");
            }
            return it->second;   // 数据已就位，直接共享
        }

        // Output 节点透传其唯一输入：直接共享上游句柄，零拷贝、零额外分配。
        // 必须先经 lookup() 按值取出——若把 values_ 内部元素直接喂给
        // emplace，rehash 会让那个引用在构造过程中悬垂。
        if (node->type == graph::NodeType::Output) {
            if (node->inputs.size() != 1) {
                throw std::runtime_error("executor: output node '" + node->name
                                         + "' must have exactly one input");
            }
            return lookup(node->inputs[0]);
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
        return publish(node->op(ins));
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

        for (const auto& node : graph.topological_sort()) {
            values_.emplace(node->id, execute_node(node));
        }
    }

    // 按依赖层并行执行。ThreadPool 由调用方注入——Executor 不持有线程池
    // （它不该拥有一个自己不一定用得上的独占资源），归属由使用者决定。
    //
    // ---------------------------------------------------------------------
    // 为什么并行是 opt-in，而不是默认路径：
    //
    // 实测（16 核，1M float）表明**并行未必更快**，收益完全取决于算子类型：
    //
    //     计算受限  sort 1MB    4 线程 3.42x   ← 接近理想
    //     内存受限  mul  4MB    4 线程 0.92x   ← 毫无收益
    //     内存受限  mul  4MB   16 线程 0.32x   ← 反而慢 3 倍
    //
    // 原因是逐元素算子本身受内存带宽限制，多核一起跑会互相抢带宽；
    // 而 sort 这类计算密集的算子才能真正吃满多核。
    //
    // Executor 无法判断一个算子属于哪一类（闭包是类型擦除的 std::function，
    // 没有任何成本信息），所以这个判断只能交给调用方。
    // ---------------------------------------------------------------------
    //
    // min_nodes_per_layer：层太窄时不值得付线程调度与同步的开销。
    void run_parallel(const graph::Graph<T>& graph,
                      ThreadPool& pool,
                      std::size_t min_nodes_per_layer = 2) {
        values_.clear();
        for (const auto& kv : inputs_) {
            values_.emplace(kv.first, kv.second);
        }

        for (const auto& layer : graph.topological_layers()) {
            if (layer.size() < min_nodes_per_layer) {
                for (const auto& node : layer) {
                    values_.emplace(node->id, execute_node(node));
                }
                continue;
            }

            // 并行阶段只往各自的槽位写，不碰 values_——多线程并发修改
            // unordered_map 是未定义行为。层结束后单线程合并。
            // 并行期间 values_ 只被读（lookup），而读是安全的。
            std::vector<tensor::TensorPtr<T>> results(layer.size());

            std::vector<std::future<void>> futures;
            futures.reserve(layer.size());
            for (std::size_t i = 0; i < layer.size(); ++i) {
                futures.push_back(pool.enqueue([this, &layer, &results, i] {
                    results[i] = execute_node(layer[i]);
                }));
            }

            // 层内屏障。节点里抛出的异常会在这里重抛。
            for (auto& f : futures) f.get();

            for (std::size_t i = 0; i < layer.size(); ++i) {
                values_.emplace(layer[i]->id, std::move(results[i]));
            }
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
