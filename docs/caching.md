# SmartEdge Caching System

## Overview

SmartEdge implements a predictive edge caching system that uses machine learning to make intelligent cache admission decisions. Unlike traditional algorithms (LRU, LFU) that react to access patterns, SmartEdge proactively predicts future reuse before admitting content to cache.

## Cache Architecture

### Cache Implementation

**Location:** `api/main.py` (in-memory OrderedDict cache)

**Data Structure:**
```python
cache = OrderedDict()
# Key: URL
# Value: {
#     "frequency": int,
#     "last_seen": int,
#     "last_seen_time": datetime,
#     "cached_at": datetime,
#     "reusable_probability": float,
#     "expected_cache_value": float,
#     "content_size": float
# }
```

### Configuration

**Cache Capacity:** 100 items (configurable)

**Admission Threshold:** 0.50 (configurable)

**Eviction Policy:** Score-based (lowest score evicted)

## Cache Admission Flow

### Step 1: Cache Hit Check

```python
if url in cache:
    # CACHE HIT
    stats["hits"] += 1
    item["frequency"] += 1
    item["last_seen"] = current_index
    item["last_seen_time"] = now
    cache.move_to_end(url)  # Mark as recently used
    return HIT response
```

### Step 2: Feature Extraction (Cache Miss)

```python
# Extract features for ML prediction
hour = now.hour
day_of_week = now.weekday()
past_frequency = request_history.get(url, {}).get("frequency", 0)
recency_seconds = (now - last_request_time).total_seconds() if url in history else 1_000_000
content_size = request.content_size
```

### Step 3: ML Prediction

```python
prediction = predict_cache_value(
    hour, day_of_week, past_frequency, recency_seconds, content_size
)
reusable_probability = prediction["reusable_probability"]
expected_cache_value = prediction["expected_cache_value"]
```

### Step 4: Admission Decision

```python
should_cache = reusable_probability >= THRESHOLD  # Default 0.50
```

### Step 5: Cache Admission

**If cache has space:**
```python
if len(cache) < CACHE_CAPACITY:
    cache[url] = {
        "frequency": 1,
        "last_seen": current_index,
        "last_seen_time": now,
        "cached_at": now,
        "reusable_probability": reusable_probability,
        "expected_cache_value": expected_cache_value,
        "content_size": content_size
    }
    stats["admissions"] += 1
```

**If cache is full:**
```python
else:
    # Find lowest-value item
    victim_url = min(cache.keys(), key=lambda url: calculate_cache_score(cache[url]))
    victim_score = calculate_cache_score(cache[victim_url])
    
    # Calculate score for new item
    new_score = calculate_cache_score({
        "reusable_probability": reusable_probability,
        "expected_cache_value": expected_cache_value,
        "frequency": 1,
        "content_size": content_size
    })
    
    # Replace only if new item has higher score
    if new_score > victim_score:
        del cache[victim_url]
        cache[url] = new_item
        stats["admissions"] += 1
        stats["evictions"] += 1
    else:
        stats["rejections"] += 1
```

### Step 6: Cache Rejection

```python
if not should_cache:
    stats["rejections"] += 1
    # Content not cached, served from origin
```

## Cache Score Calculation

The cache score determines which items are evicted when the cache is full.

### Score Formula

```python
def calculate_cache_score(item):
    probability = item["reusable_probability"]
    expected_value = item["expected_cache_value"]
    frequency = max(item["frequency"], 1)
    current_index = stats["total_requests"]
    last_seen = item["last_seen"]
    age = max(current_index - last_seen, 1)
    size = max(item["content_size"], 1.0)
    
    # Frequency component (logarithmic)
    frequency_factor = math.log1p(frequency)
    
    # Recency component (inverse of age)
    recency_factor = 1.0 / (1.0 + math.log1p(age))
    
    # Size penalty (larger files penalized)
    size_factor = 1.0 / (1.0 + math.log1p(size) / 10.0)
    
    # Combined score
    score = (
        expected_value *
        probability *
        frequency_factor *
        recency_factor *
        size_factor
    )
    
    return score
```

### Score Components

1. **Expected Cache Value:** ML prediction of reuse value (0-4)
2. **Reusable Probability:** Probability of future reuse (0-1)
3. **Frequency Factor:** Logarithmic frequency (higher = better)
4. **Recency Factor:** Inverse of age (more recent = better)
5. **Size Factor:** Inverse of size (smaller = better)

## Cache Statistics

### Tracked Metrics

```python
stats = {
    "total_requests": 0,        # Total requests processed
    "hits": 0,                  # Cache hits
    "misses": 0,                # Cache misses
    "admissions": 0,            # Items admitted to cache
    "rejections": 0,             # Items rejected (below threshold)
    "evictions": 0,             # Items evicted from cache
    "ml_predictions": 0,        # Total ML predictions made
    "total_response_time_ms": 0.0  # Cumulative response time
}
```

### Derived Metrics

**Hit Rate:**
```python
hit_rate = (hits / total_requests) * 100
```

**Miss Rate:**
```python
miss_rate = (misses / total_requests) * 100
```

**Cache Utilization:**
```python
utilization = (cache_size / cache_capacity) * 100
```

**Average Response Time:**
```python
avg_response_time = total_response_time_ms / total_requests
```

## Comparison with Traditional Algorithms

### LRU (Least Recently Used)

**Policy:** Evict the least recently used item

**Advantages:**
- Simple to implement
- Low overhead
- Good for temporal locality

**Disadvantages:**
- No prediction of future reuse
- Vulnerable to one-time large file scans
- Cannot adapt to access patterns

### LFU (Least Frequently Used)

**Policy:** Evict the least frequently used item

**Advantages:**
- Accounts for frequency
- Good for stable access patterns

**Disadvantages:**
- No temporal awareness
- Vulnerable to frequency bursts
- Cold start problem

### SmartEdge

**Policy:** ML prediction + score-based eviction

**Advantages:**
- Predicts future reuse before caching
- Considers multiple factors (probability, frequency, recency, size)
- Adapts to access patterns via ML
- Explainable decisions

**Disadvantages:**
- Higher computational overhead (ML inference)
- Requires trained model
- More complex implementation

## Benchmark Results

### NASA Trace (10,000 requests)

**At capacity 100:**
- LRU: 59.54% hit rate
- LFU: 65.85% hit rate
- SmartEdge: 68.11% hit rate
- Improvement vs LFU: +2.26 percentage points

**At capacity 25 (constrained):**
- LRU: 31.66% hit rate
- LFU: 44.45% hit rate
- SmartEdge: 47.26% hit rate
- Improvement vs LFU: +2.81 percentage points

### Key Findings

1. SmartEdge outperforms both LRU and LFU across all tested capacities
2. Advantage is most significant at constrained cache capacities
3. Advantage decreases as cache capacity increases (diminishing returns)
4. Best admission threshold: 0.1 (for this dataset)

## Adaptive Threshold

SmartEdge supports configurable admission thresholds. The optimal threshold depends on:

- Cache capacity
- Access pattern characteristics
- Content size distribution
- Performance objectives

### Threshold Trade-offs

**Low Threshold (e.g., 0.1):**
- More admissions
- Higher hit rate
- More evictions
- Higher computational overhead

**High Threshold (e.g., 0.9):**
- Fewer admissions
- Lower hit rate
- Fewer evictions
- Lower computational overhead

### Threshold Experiments

**Location:** `model/adaptive_threshold_results_v3.csv`

Contains results for thresholds 0.1-0.9 across capacities 25-500.

## Cache Operations API

### Get Cache Contents
```
GET /cache
```

### Get Cache Statistics
```
GET /cache/stats
```

### Remove Specific Item
```
DELETE /cache/item?url={url}
```

### Reset Cache
```
POST /cache/reset
```

### Process Request
```
POST /request
{
  "url": "/images/example.jpg",
  "content_size": 45000
}
```

## Performance Considerations

### Time Complexity

- **Cache Hit:** O(1) - OrderedDict lookup
- **Cache Miss:** O(1) + ML inference + O(n) for eviction (n = cache size)
- **ML Inference:** O(t) where t = number of trees (typically 100)
- **Score Calculation:** O(1) per item

### Space Complexity

- **Cache Storage:** O(capacity × average_item_size)
- **Request History:** O(unique_urls)
- **ML Model:** O(model_size) ~690MB

### Optimization Opportunities

1. **Batch Predictions:** Process multiple requests in a single ML call
2. **Model Quantization:** Reduce model size with quantization
3. **Approximate Nearest Neighbor:** Use ANN for eviction candidate selection
4. **Threshold Tuning:** Dynamically adjust threshold based on hit rate
5. **Cache Warming:** Pre-populate cache with high-probability items
