#ifndef ADVANCED_ITERATOR_H
#define ADVANCED_ITERATOR_H

#include <vector>
#include <list>
#include <deque>
#include <set>
#include <array>
#include <stdexcept>
#include <iterator>
#include <memory>
#include <type_traits>

// 自定义迭代器异常
class StopIteration : public std::exception {
public:
    const char* what() const noexcept override {
        return "StopIteration";
    }
};

// 通用迭代器，支持多种容器
template<typename Container>
class AdvancedIterator {
private:
    using iterator_type = typename Container::const_iterator;
    iterator_type current;
    iterator_type end;

public:
    // 构造函数
    AdvancedIterator(const Container& container)
        : current(std::begin(container)), end(std::end(container)) {}

    // 获取下一个元素
    auto next() -> typename Container::value_type {
        if (current == end) {
            throw StopIteration();
        }
        return *current++;
    }

    // 检查是否还有下一个元素
    bool has_next() const {
        return current != end;
    }

    // 获取剩余元素数量
    size_t remaining() const {
        return std::distance(current, end);
    }

    // 查看下一个元素但不移动迭代器
    auto peek() const -> typename Container::value_type {
        if (current == end) {
            throw StopIteration();
        }
        return *current;
    }

    // 跳过指定数量的元素
    void skip(size_t count = 1) {
        for (size_t i = 0; i < count && current != end; ++i) {
            ++current;
        }
    }
};

// 通用 iter 函数
template<typename Container>
AdvancedIterator<Container> advanced_iter(const Container& container) {
    return AdvancedIterator<Container>(container);
}

// 通用 next 函数 - 修正版本
template<typename Container>
auto advanced_next(AdvancedIterator<Container>& iterator) -> typename Container::value_type {
    return iterator.next();
}

template<typename Container>
auto advanced_next(AdvancedIterator<Container>& iterator, 
                   const typename Container::value_type& default_value) -> typename Container::value_type {
    if (!iterator.has_next()) {
        return default_value;
    }
    return iterator.next();
}

// 范围迭代器 - 用于数值范围
class RangeIterator {
private:
    long long current;
    long long end;
    long long step;

public:
    RangeIterator(long long start, long long stop, long long step_size = 1)
        : current(start), end(stop), step(step_size) {}

    long long next() {
        if ((step > 0 && current >= end) || (step < 0 && current <= end)) {
            throw StopIteration();
        }
        long long result = current;
        current += step;
        return result;
    }

    bool has_next() const {
        return (step > 0 && current < end) || (step < 0 && current > end);
    }

    size_t remaining() const {
        if ((step > 0 && current >= end) || (step < 0 && current <= end)) {
            return 0;
        }
        return std::abs((end - current) / step);
    }
};

// 范围生成函数
inline RangeIterator range(long long stop) {
    return RangeIterator(0, stop, 1);
}

inline RangeIterator range(long long start, long long stop) {
    return RangeIterator(start, stop, 1);
}

inline RangeIterator range(long long start, long long stop, long long step) {
    return RangeIterator(start, stop, step);
}

#endif // ADVANCED_ITERATOR_H