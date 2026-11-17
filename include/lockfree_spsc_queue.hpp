#ifndef LOCKFREE_SPSC_QUEUE_HPP
#define LOCKFREE_SPSC_QUEUE_HPP

#include <vector>
#include <atomic>
#include <cstddef>

template<typename T, size_t Capacity = 2048>
class LockFreeSPSCQueue {
public:
    LockFreeSPSCQueue() : head_(0), tail_(0) {
        buffer_.resize(Capacity);
    }

    bool push(const T& item) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % Capacity;
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false;
        }
        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        item = buffer_[current_head];
        head_.store((current_head + 1) % Capacity, std::memory_order_release);
        return true;
    }

private:
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    std::vector<T> buffer_;
};

#endif // LOCKFREE_SPSC_QUEUE_HPP
