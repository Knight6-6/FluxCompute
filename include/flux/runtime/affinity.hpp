#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <string>

#if defined(__linux__)
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>

#ifndef MPOL_DEFAULT
#define MPOL_DEFAULT    0
#endif
#ifndef MPOL_PREFERRED
#define MPOL_PREFERRED  1
#endif
#ifndef MPOL_BIND
#define MPOL_BIND       2
#endif
#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif
#ifndef MPOL_LOCAL
#define MPOL_LOCAL      4
#endif
#endif

namespace flux::runtime {

/**
 * @brief 获取当前线程正在执行的 CPU 逻辑核心 ID
 */
inline int get_current_cpu() {
#if defined(__linux__)
    return sched_getcpu();
#else
    return 0;
#endif
}

/**
 * @brief 获取当前线程正在运行的 NUMA 节点 ID
 */
inline int get_current_numa_node() {
#if defined(__linux__)
    unsigned int cpu = 0, node = 0;
#if defined(SYS_getcpu)
    if (syscall(SYS_getcpu, &cpu, &node, nullptr) == 0) {
        return static_cast<int>(node);
    }
#endif
    return 0;
#else
    return 0;
#endif
}

/**
 * @brief 将当前线程绑定到指定的单个 CPU 逻辑核心
 *
 * @param cpu_id 目标 CPU 核心编号 (0 ~ N-1)
 * @return true 成功, false 失败
 */
inline bool set_thread_affinity(int cpu_id) {
    if (cpu_id < 0) return true;
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    return (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0);
#else
    (void)cpu_id;
    return false;
#endif
}

/**
 * @brief 将当前线程绑定到指定的 CPU 核心集合 (掩码)
 */
inline bool set_thread_affinity(const std::vector<int>& cpu_ids) {
    if (cpu_ids.empty()) return true;
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    bool any_valid = false;
    for (int cpu : cpu_ids) {
        if (cpu >= 0) {
            CPU_SET(cpu, &cpuset);
            any_valid = true;
        }
    }
    if (!any_valid) return true;
    return (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0);
#else
    (void)cpu_ids;
    return false;
#endif
}

/**
 * @brief 绑定当前线程的内存分配策略至指定 NUMA 节点
 *
 * 设定后，由该线程分配的堆内存 (malloc / posix_memalign / new) 将强制锁定在该 NUMA 节点的本地内存中，
 * 彻底消除双路/多路 CPU 跨 Socket 访问远端内存带来的延迟惩罚。
 *
 * @param numa_node NUMA 节点号 (0 ~ max_nodes-1)
 * @return true 成功, false 失败
 */
inline bool set_thread_numa_node(int numa_node) {
    if (numa_node < 0) return true;
#if defined(__linux__) && defined(SYS_set_mempolicy)
    unsigned long nodemask = (1UL << numa_node);
    unsigned long maxnode = sizeof(unsigned long) * 8;
    // 优先尝试严格绑定 MPOL_BIND，若由于系统限制失败则尝试 MPOL_PREFERRED 优先本地
    if (syscall(SYS_set_mempolicy, MPOL_BIND, &nodemask, maxnode) == 0) {
        return true;
    }
    return (syscall(SYS_set_mempolicy, MPOL_PREFERRED, &nodemask, maxnode) == 0);
#else
    (void)numa_node;
    return false;
#endif
}

/**
 * @brief 同时完成线程 CPU 绑核与对应 NUMA 内存节点绑定
 *
 * @param cpu_id 目标 CPU 核心 ID
 * @param numa_node 目标 NUMA 节点 ID。若传 -1，则自动推导并锁定该核心所属的本地 NUMA 节点。
 */
inline bool bind_thread(int cpu_id, int numa_node = -1) {
    bool ok = true;
    if (cpu_id >= 0) {
        ok = set_thread_affinity(cpu_id) && ok;
    }
    if (numa_node >= 0) {
        ok = set_thread_numa_node(numa_node) && ok;
    } else if (cpu_id >= 0) {
        int auto_node = get_current_numa_node();
        if (auto_node >= 0) {
            set_thread_numa_node(auto_node);
        }
    }
    return ok;
}

} // namespace flux::runtime
