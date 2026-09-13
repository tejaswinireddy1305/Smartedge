# SmartEdge ML Pipeline

## Overview

SmartEdge uses a Random Forest classifier to predict content reuse behavior. The ML pipeline transforms raw request data into features, trains a model, and uses the model to make cache admission decisions.

## Dataset

**Source:** NASA web server access logs

**Location:** `dataset/processed/final_ml_dataset.csv`

**Size:** ~115MB (processed dataset)

**Features:**
- `timestamp` - Request timestamp
- `url` - Resource URL
- `content_size` - Size in bytes
- `hour` - Hour of day (0-23)
- `day_of_week` - Day of week (0-6, Monday=0)
- `past_frequency` - Number of previous requests for this URL
- `recency_seconds` - Time since last request in seconds

## Feature Engineering

### 1. Temporal Features

**Hour (0-23):**
- Captures diurnal access patterns
- Peak hours may indicate higher reuse probability

**Day of Week (0-6):**
- Captures weekly access patterns
- Weekdays vs weekends may have different patterns

### 2. Frequency Features

**Past Frequency:**
- Count of previous requests for the same URL
- Higher frequency indicates higher reuse probability
- Calculated dynamically during request processing

### 3. Recency Features

**Recency Seconds:**
- Time elapsed since last request for the same URL
- Lower recency (more recent) indicates higher reuse probability
- Large values (e.g., 999999) indicate first-time requests

### 4. Content Features

**Content Size:**
- Size of the resource in bytes
- Larger files may have different caching behavior
- Influences cache score calculation

## Model Training

### Model Type

**Algorithm:** Random Forest Classifier

**Library:** scikit-learn

**Configuration:**
- Number of trees: 100 (default)
- Max depth: None (default)
- Random state: 42 (for reproducibility)

### Training Process

**Location:** `ml/train_model.py` or `api/train_smartedge_v3.py`

**Steps:**
1. Load processed dataset
2. Extract features: `hour`, `day_of_week`, `past_frequency`, `recency_seconds`, `content_size`
3. Create target variable based on future reuse (derived from request patterns)
4. Split data: 80% training, 20% testing
5. Train Random Forest classifier
6. Evaluate model accuracy
7. Save model to `model/smartedge_v3_random_forest.pkl`

### Target Variable Construction

The target variable is constructed by analyzing future request patterns:
- **NO_REUSE (0):** Content never requested again
- **LOW_REUSE (1):** Content requested 1-2 times in future
- **MEDIUM_REUSE (2):** Content requested 3-5 times in future
- **HIGH_REUSE (3):** Content requested 6-10 times in future
- **VERY_HIGH_REUSE (4):** Content requested 10+ times in future

## Model Inference

### Prediction Process

**Location:** `api/main.py` - `predict_cache_value()` function

**Input:**
```python
{
    "hour": 12,
    "day_of_week": 2,
    "past_frequency": 20,
    "recency_seconds": 10,
    "content_size": 1839
}
```

**Steps:**
1. Create DataFrame with feature columns
2. Call `model.predict_proba(features)`
3. Extract class probabilities
4. Calculate reusable probability (sum of classes 1-4)
5. Calculate expected cache value (weighted sum)
6. Return prediction with metadata

**Output:**
```python
{
    "predicted_class": 3,
    "predicted_label": "HIGH_REUSE",
    "reusable_probability": 0.8234,
    "expected_cache_value": 3.2145,
    "class_probabilities": {
        "0": 0.02,
        "1": 0.05,
        "2": 0.10,
        "3": 0.75,
        "4": 0.08
    }
}
```

## Cache Admission Decision

### Threshold-Based Admission

**Default Threshold:** 0.50

**Logic:**
```python
if reusable_probability >= THRESHOLD:
    # Admit to cache
    if cache_has_space:
        admit_immediately()
    else:
        if new_score > victim_score:
            evict_and_admit()
        else:
            reject()
else:
    # Do not cache
    reject()
```

### Expected Cache Value

Used for cache scoring and eviction decisions:

```python
expected_value = sum(
    probability[class] * class_value
    for class in classes
)

class_values = {
    0: 0.0,   # NO_REUSE
    1: 1.0,   # LOW_REUSE
    2: 2.0,   # MEDIUM_REUSE
    3: 3.0,   # HIGH_REUSE
    4: 4.0    # VERY_HIGH_REUSE
}
```

## Model Performance

### Feature Importance

**Location:** `model/smartedge_v3_feature_importance.csv`

Typical importance ranking:
1. `past_frequency` - Most important
2. `recency_seconds` - Second most important
3. `hour` - Moderate importance
4. `day_of_week` - Lower importance
5. `content_size` - Least important

### Accuracy

The model achieves approximately 75-80% accuracy on test data, depending on the specific dataset split and threshold configuration.

## Model File

**Location:** `model/smartedge_v3_random_forest.pkl`

**Size:** ~690MB (includes trained forest and feature information)

**Loading:**
```python
import joblib

model = joblib.load("model/smartedge_v3_random_forest.pkl")
model.n_jobs = 1  # Set to 1 for thread safety
```

## Retraining

To retrain the model with new data:

1. Prepare new dataset with same feature schema
2. Run training script: `python ml/train_model.py`
3. Backup old model file
4. Replace with new model file
5. Restart API server

## Online Learning

The current implementation does not support online learning. The model is trained offline and deployed as a static model. For production use, consider:

- Periodic retraining with new data
- A/B testing new model versions
- Monitoring prediction drift
- Feature importance tracking over time
