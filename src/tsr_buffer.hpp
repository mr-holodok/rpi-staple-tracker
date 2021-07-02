#ifndef THREAD_SAFE_REWRITABLE_BUFFER_H
#define THREAD_SAFE_REWRITABLE_BUFFER_H

#include <exception>
#include <memory>
#include <mutex>

// thread safe buffer that rewrites old data and returns a last
// added value without poping that value  
template <class T> 
class TSRBuffer {
public:
	explicit TSRBuffer(size_t size) :
		_buf(std::unique_ptr<T[]>(new T[size])),
		_maxSize(size) { }
	
	void put(T item) {
		std::lock_guard<std::mutex> lock(_mutex);
        _buf[_nextEmptyCellIndex] = item;
		_empty = false;
        _nextEmptyCellIndex = (_nextEmptyCellIndex + 1) % _maxSize;
	} 
	
	T get() {
        std::lock_guard<std::mutex> lock(_mutex); 
        if (_empty)
            throw std::runtime_error("Buffer is empty!");

        // + case when _nextEmptyCellIndex is 0
        return _buf[(_maxSize + _nextEmptyCellIndex - 1) % _maxSize];
    }

	void reset() {
		std::lock_guard<std::mutex> lock(_mutex);
        _nextEmptyCellIndex = 0;
		_empty = true;
	}
	
	bool empty() const {
		return _empty;
	}
	
	size_t capacity() const {
		return _maxSize;
	}
	
private:
	std::mutex _mutex;
	std::unique_ptr<T[]> _buf;
    size_t _nextEmptyCellIndex {0};
	const size_t _maxSize;
	bool _empty {true};
};

#endif
