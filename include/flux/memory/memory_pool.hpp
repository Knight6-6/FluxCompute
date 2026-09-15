#pragma once
#include <flux/memory/aligned_allocator.hpp>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstddef>

namespace flux::memory {

class MemoryPool {
private:
    // 按字节大小（Size Class）组织空闲对齐内存块
    std::unordered_map<std::size_t, std::vector<void*>> free_blocks_;
    std::mutex mutex_;
    static constexpr std::size_t Alignment = 64;

public:
    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }

    void* allocate(std::size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& blocks = free_blocks_[bytes];
        if (!blocks.empty()) {
            void* ptr = blocks.back();
            blocks.pop_back();
            return ptr;
        }
        
        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(bytes, Alignment);
#else
        if (posix_memalign(&ptr, Alignment, bytes) != 0) ptr = nullptr;
#endif
        return ptr;
    }

    void deallocate(void* ptr, std::size_t bytes) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex_);
        free_blocks_[bytes].push_back(ptr);
    }

    ~MemoryPool() {
        for (auto& [bytes, blocks] : free_blocks_) {
            for (void* ptr : blocks) {
#if defined(_MSC_VER)
                _aligned_free(ptr);
#else
                free(ptr);
#endif
            }
        }
    }
};

}