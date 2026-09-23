#pragma once
#include <flux/runtime/affinity.hpp>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <cstddef>
#include <stdexcept>

namespace flux::runtime {

/**
 * @brief 单个工作线程的亲和性与 NUMA 绑定配置
 */
struct ThreadConfig {
    int cpu_id = -1;       // -1 表示由 OS 默认调度; >= 0 表示硬绑定到指定 CPU 逻辑核心
    int numa_node = -1;    // -1 表示由 cpu_id 自动感知 NUMA 节点; >= 0 表示强制绑定到特定 NUMA 内存节点
};

class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_ = false;

    void init_workers(const std::vector<ThreadConfig>& configs) {
        for (const auto& cfg : configs) {
            workers_.emplace_back([this, cfg] {
                // 工作线程启动时执行硬绑核与 NUMA 内存策略锁定
                bind_thread(cfg.cpu_id, cfg.numa_node);

                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->cv_.wait(lock, [this] { return this->stop_ || !this->tasks_.empty(); });
                        if (this->stop_ && this->tasks_.empty()) return;
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

public:
    /**
     * @brief 默认构造函数：启动指定数量的工作线程 (不主动绑核，交由 OS 调度)
     */
    explicit ThreadPool(std::size_t threads = std::thread::hardware_concurrency()) {
        // C++ 标准规定 hardware_concurrency() 在无法探测核心数时允许返回 0
        // （例如无权限读取 sysfs/affinity 的容器、沙箱）。
        // 0 线程会导致入队任务永远无法被 worker 消费，在 future.get() 处产生死锁挂起。
        if (threads == 0) threads = 1;
        std::vector<ThreadConfig> configs(threads);
        init_workers(configs);
    }

    /**
     * @brief 绑核构造函数：传入指定绑定的 CPU 核心列表
     *
     * 创建 cpu_cores.size() 个工作线程，第 i 个 worker 线程绑定到 cpu_cores[i]。
     *
     * @param cpu_cores 目标 CPU 核心 ID 列表 (例如 {6, 7, 8, 9})
     * @param auto_bind_numa 是否自动将各线程的内存分配策略锁定在其所在核心的 NUMA 节点上 (默认 true)
     */
    explicit ThreadPool(const std::vector<int>& cpu_cores, bool auto_bind_numa = true) {
        std::size_t n = cpu_cores.empty() ? 1 : cpu_cores.size();
        std::vector<ThreadConfig> configs;
        configs.reserve(n);
        if (cpu_cores.empty()) {
            configs.push_back(ThreadConfig{-1, -1});
        } else {
            for (int core : cpu_cores) {
                configs.push_back(ThreadConfig{core, auto_bind_numa ? -1 : -2});
            }
        }
        init_workers(configs);
    }

    /**
     * @brief 精细化配置构造函数：为每个工作线程精准指定 cpu_id 和 numa_node
     */
    explicit ThreadPool(const std::vector<ThreadConfig>& configs) {
        if (configs.empty()) {
            std::vector<ThreadConfig> default_cfg(1);
            init_workers(default_cfg);
        } else {
            init_workers(configs);
        }
    }

    std::size_t size() const {
        return workers_.size();
    }

    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) throw std::runtime_error("Enqueue on stopped ThreadPool");
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
};

} // namespace flux::runtime