# SmartEdge API Documentation

## Base URL

```
http://127.0.0.1:8001
```

## Endpoints

### 1. Root

**GET /**

Returns API status information.

**Response:**
```json
{
  "status": "online",
  "project": "SmartEdge",
  "version": "3.0",
  "message": "SmartEdge 3.0 API is running"
}
```

### 2. Health Check

**GET /health**

Checks system health and model availability.

**Response:**
```json
{
  "status": "healthy",
  "model": "SmartEdge 3.0",
  "model_exists": true,
  "dataset_exists": true,
  "results_exists": true,
  "cache_capacity": 100,
  "current_cache_size": 0
}
```

### 3. Cache Statistics

**GET /cache/stats**

Returns live cache performance metrics.

**Response:**
```json
{
  "status": "success",
  "total_requests": 1248,
  "hits": 892,
  "misses": 356,
  "hit_rate": 71.5,
  "miss_rate": 28.5,
  "admissions": 45,
  "rejections": 311,
  "evictions": 12,
  "ml_predictions": 356,
  "cache_capacity": 100,
  "cache_size": 45,
  "cache_utilization": 45.0,
  "average_response_time_ms": 2.345
}
```

### 4. Cache Contents

**GET /cache**

Returns currently cached items.

**Response:**
```json
{
  "status": "success",
  "size": 45,
  "capacity": 100,
  "items": [
    {
      "url": "/images/banner.jpg",
      "frequency": 5,
      "reusable_probability": 0.8234,
      "expected_cache_value": 3.2145,
      "content_size": 45000,
      "cached_at": "2024-01-15T10:30:45",
      "score": 2.4567
    }
  ]
}
```

### 5. Remove Cache Item

**DELETE /cache/item?url={url}**

Removes a specific item from cache.

**Parameters:**
- `url` (query parameter): URL of item to remove

**Response:**
```json
{
  "status": "success",
  "message": "Removed /images/banner.jpg from cache",
  "cache_size": 44
}
```

### 6. Reset Cache

**POST /cache/reset**

Clears all cache contents and resets statistics.

**Response:**
```json
{
  "status": "success",
  "message": "SmartEdge cache and statistics reset",
  "cache_size": 0
}
```

### 7. ML Prediction

**POST /predict**

Runs ML prediction for cache admission decision.

**Request Body:**
```json
{
  "hour": 12,
  "day_of_week": 2,
  "past_frequency": 20,
  "recency_seconds": 10,
  "content_size": 1839
}
```

**Response:**
```json
{
  "status": "success",
  "predicted_class": 3,
  "predicted_label": "HIGH_REUSE",
  "reusable_probability": 0.8234,
  "reusable_probability_percent": 82.34,
  "expected_cache_value": 3.2145,
  "threshold": 0.50,
  "cache_decision": "CACHE",
  "class_probabilities": {
    "0": 0.02,
    "1": 0.05,
    "2": 0.10,
    "3": 0.75,
    "4": 0.08
  }
}
```

### 8. Process Request

**POST /request**

Simulates a resource request through the SmartEdge caching engine.

**Request Body:**
```json
{
  "url": "/images/example.jpg",
  "content_size": 45000
}
```

**Response (Cache Hit):**
```json
{
  "status": "success",
  "url": "/images/example.jpg",
  "cache_status": "HIT",
  "message": "Resource served from SmartEdge cache",
  "source": "EDGE_CACHE",
  "prediction": "Already cached",
  "reusable_probability": 0.8234,
  "expected_cache_value": 3.2145,
  "response_time_ms": 1.234,
  "cache_size": 45,
  "cache_capacity": 100
}
```

**Response (Cache Miss):**
```json
{
  "status": "success",
  "url": "/images/new.jpg",
  "cache_status": "MISS",
  "message": "Resource was not in cache",
  "source": "ORIGIN_SIMULATION",
  "ml_prediction": {
    "predicted_class": 2,
    "predicted_label": "MEDIUM_REUSE",
    "reusable_probability": 0.6234,
    "reusable_probability_percent": 62.34,
    "expected_cache_value": 2.1234
  },
  "cache_decision": {
    "threshold": 0.50,
    "admitted": true,
    "decision": "CACHE",
    "evicted_url": null
  },
  "response_time_ms": 5.678,
  "cache_size": 46,
  "cache_capacity": 100
}
```

### 9. Benchmark Summary

**GET /summary**

Returns offline benchmark comparison results.

**Response:**
```json
{
  "status": "success",
  "total_requests": 100000,
  "simulation_requests": 100000,
  "best_algorithm": "SmartEdge",
  "smartedge_hit_rate": 68.11,
  "smartedge_miss_rate": 31.89,
  "smartedge_evictions": 1227,
  "smartedge_proactive_insertions": 1327,
  "smartedge_vs_lru_percentage_points": 8.57,
  "smartedge_vs_lfu_percentage_points": 2.26,
  "algorithms": {
    "LRU": {
      "requests": 100000,
      "hits": 63500,
      "misses": 36500,
      "hit_rate": 63.5,
      "miss_rate": 36.5,
      "evictions": 36400,
      "proactive_insertions": 0
    },
    "LFU": {
      "requests": 100000,
      "hits": 67707,
      "misses": 32293,
      "hit_rate": 67.707,
      "miss_rate": 32.293,
      "evictions": 32193,
      "proactive_insertions": 0
    },
    "SmartEdge": {
      "requests": 100000,
      "hits": 60737,
      "misses": 39263,
      "hit_rate": 60.737,
      "miss_rate": 39.263,
      "evictions": 0,
      "proactive_insertions": 65
    }
  }
}
```

### 10. AI Agent

**POST /agent/ask**

Queries the multi-agent AI system.

**Request Body:**
```json
{
  "question": "Why does SmartEdge perform better than LFU?"
}
```

**Response:**
```json
{
  "question": "Why does SmartEdge perform better than LFU?",
  "selected_agent": "Performance Agent",
  "result": {
    "question": "Why does SmartEdge perform better than LFU?",
    "answer": "SmartEdge achieves a 2.26 percentage point improvement over LFU at capacity 100...",
    "benchmark_source": "final_cache_comparison.csv",
    "findings": [
      "SmartEdge hit rate: 68.11%",
      "LFU hit rate: 65.85%",
      "SmartEdge difference versus LFU: 2.26 percentage points"
    ]
  },
  "live_context": {
    "health": {
      "status": "healthy",
      "model_exists": true,
      "cache_capacity": 100,
      "current_cache_size": 45
    },
    "cache_stats": {
      "total_requests": 1248,
      "hit_rate": 71.5,
      "cache_size": 45
    }
  }
}
```

## Error Responses

All endpoints may return error responses:

```json
{
  "status": "error",
  "message": "Error description"
}
```

Common error scenarios:
- Model file not found
- Dataset file not found
- Invalid input parameters
- Backend service unavailable

## Rate Limiting

No rate limiting is currently implemented. The API is designed for demonstration and research purposes.

## CORS

CORS is enabled for all origins (`allow_origins=["*"]`) to facilitate frontend development.

## Configuration

### Cache Configuration
- **Capacity:** 100 items
- **Admission Threshold:** 0.50

### Model Configuration
- **Model File:** `model/smartedge_v3_random_forest.pkl`
- **Features:** hour, day_of_week, past_frequency, recency_seconds, content_size
- **Classes:** 5 (NO_REUSE, LOW_REUSE, MEDIUM_REUSE, HIGH_REUSE, VERY_HIGH_REUSE)

### LLM Configuration
- **Provider:** Ollama
- **Model:** llama3.2:3b
- **Purpose:** Grounded explanation generation
