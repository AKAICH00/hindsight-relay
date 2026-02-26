package main

import (
	"sync"
)

type QueueItem struct {
	Collection string
	VectorID   string
	Vector     []float32
	Payload    map[string]any
	Content    string
	Hash       string
}

type RetryQueue struct {
	mu    sync.Mutex
	items []QueueItem
	max   int
}

func NewRetryQueue(max int) *RetryQueue {
	return &RetryQueue{
		items: make([]QueueItem, 0, max),
		max:   max,
	}
}

func (q *RetryQueue) Enqueue(item QueueItem) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) >= q.max {
		return false
	}
	q.items = append(q.items, item)
	return true
}

func (q *RetryQueue) PopAll() []QueueItem {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return nil
	}
	out := make([]QueueItem, len(q.items))
	copy(out, q.items)
	q.items = q.items[:0]
	return out
}

func (q *RetryQueue) PushFront(items []QueueItem) {
	if len(items) == 0 {
		return
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	space := q.max - len(q.items)
	if space <= 0 {
		return
	}
	if len(items) > space {
		items = items[:space]
	}
	combined := make([]QueueItem, 0, len(items)+len(q.items))
	combined = append(combined, items...)
	combined = append(combined, q.items...)
	q.items = combined
}

func (q *RetryQueue) Depth() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items)
}

