import math
from collections import OrderedDict, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"D:\smart edge\dataset\processed\clean_requests.csv"

MODEL_PATH = r"D:\smart edge\model\smartedge_random_forest.pkl"

REQUEST_LIMIT = 10000

CACHE_CAPACITY = 100

SMARTEDGE_THRESHOLD = 0.10


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading SmartEdge ML model...")

model = joblib.load(MODEL_PATH)

print("Model:", type(model).__name__)

print(
    "Features:",
    list(model.feature_names_in_)
)


# ============================================================
# LOAD REAL REQUEST DATA
# ============================================================

print("\nLoading NASA request dataset...")

df = pd.read_csv(DATASET_PATH)

# Convert timestamp
df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

# Remove invalid URLs
df = df[
    df["url"].notna()
].copy()

# Sort chronologically
df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

# Limit experiment
df = df.head(
    REQUEST_LIMIT
).copy()

print(
    "Requests loaded:",
    len(df)
)


# ============================================================
# COMMON REQUEST STREAM
# ============================================================

requests = []

for _, row in df.iterrows():

    requests.append({
        "timestamp": row["timestamp"],
        "url": row["url"],
        "size": int(
            row["response_bytes"]
        )
    })


# ============================================================
# LRU
# ============================================================

def run_lru(requests, capacity):

    cache = OrderedDict()

    hits = 0
    misses = 0
    evictions = 0

    for request in requests:

        url = request["url"]

        if url in cache:

            hits += 1

            cache.move_to_end(url)

        else:

            misses += 1

            if len(cache) >= capacity:

                cache.popitem(
                    last=False
                )

                evictions += 1

            cache[url] = request["size"]

    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "hit_ratio": hits / len(requests),
        "cache_size": len(cache)
    }


# ============================================================
# LFU
# ============================================================

def run_lfu(requests, capacity):

    cache = set()

    frequency = defaultdict(int)

    last_used = {}

    hits = 0
    misses = 0
    evictions = 0

    for index, request in enumerate(requests):

        url = request["url"]

        frequency[url] += 1

        if url in cache:

            hits += 1

        else:

            misses += 1

            if len(cache) >= capacity:

                victim = min(
                    cache,
                    key=lambda x: (
                        frequency[x],
                        last_used.get(
                            x,
                            -1
                        )
                    )
                )

                cache.remove(
                    victim
                )

                evictions += 1

            cache.add(url)

        last_used[url] = index

    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "hit_ratio": hits / len(requests),
        "cache_size": len(cache)
    }


# ============================================================
# SMARTEDGE
# ============================================================

def run_smartedge(
    df,
    features,
    model,
    capacity,
    threshold=0.5
):
    """
    SmartEdge 3.0 predictive caching.

    Parameters
    ----------
    df : pandas.DataFrame
        Request dataset.

    features : pandas.DataFrame
        ML feature dataframe.

    model : trained ML model
        Random Forest model.

    capacity : int
        Maximum cache capacity.

    threshold : float
        Minimum reusable probability required
        for cache admission.

    Returns
    -------
    dict
        Cache experiment statistics.
    """

    print("\nGenerating SmartEdge 3.0 predictions...")

    # ============================================================
    # FEATURES
    # ============================================================

    expected_features = [
        "hour",
        "day_of_week",
        "past_frequency",
        "recency_seconds",
        "content_size"
    ]

    X = features[expected_features].astype(float)

    # ============================================================
    # ML PREDICTIONS
    # ============================================================

    probabilities = model.predict_proba(X)

    classes = [int(c) for c in model.classes_]

    # ============================================================
    # CONVERT MODEL OUTPUT TO CACHE VALUE
    # ============================================================

    class_values = {
        0: 0.0,
        1: 1.0,
        2: 2.0,
        3: 3.0,
        4: 4.0
    }

    expected_values = np.zeros(len(X), dtype=float)
    reusable_probabilities = np.zeros(len(X), dtype=float)

    for column_index, cls in enumerate(classes):

        value = class_values.get(cls, 0.0)

        expected_values += (
            probabilities[:, column_index] * value
        )

        if cls > 0:

            reusable_probabilities += (
                probabilities[:, column_index]
            )

    print("Predictions completed.")

    # ============================================================
    # PREPARE REQUEST DATA
    # ============================================================

    urls = df["url"].to_numpy()

    response_bytes = df["response_bytes"].to_numpy()

    total_requests = len(df)

    # ============================================================
    # CACHE
    # ============================================================

    cache = {}

    hits = 0
    misses = 0

    admissions = 0
    rejected = 0
    evictions = 0

    total_probability = 0.0
    total_expected_value = 0.0

    # ============================================================
    # PROCESS REQUESTS
    # ============================================================

    for index in range(total_requests):

        url = urls[index]

        probability = float(
            reusable_probabilities[index]
        )

        expected_value = float(
            expected_values[index]
        )

        total_probability += probability
        total_expected_value += expected_value

        # ========================================================
        # CACHE HIT
        # ========================================================

        if url in cache:

            hits += 1

            item = cache[url]

            item["frequency"] += 1
            item["last_seen"] = index

            item["probability"] = probability
            item["expected_value"] = expected_value

            continue

        # ========================================================
        # CACHE MISS
        # ========================================================

        misses += 1

        # ========================================================
        # ADMISSION
        # ========================================================

        if probability < threshold:

            rejected += 1

            continue

        # ========================================================
        # NEW OBJECT
        # ========================================================

        new_size = max(
            float(response_bytes[index]),
            1.0
        )

        # ========================================================
        # CACHE HAS SPACE
        # ========================================================

        if len(cache) < capacity:

            cache[url] = {

                "frequency": 1,

                "last_seen": index,

                "probability": probability,

                "expected_value": expected_value,

                "size": new_size
            }

            admissions += 1

            continue

        # ========================================================
        # CACHE FULL
        # ========================================================

        def cache_score(item):

            data = cache[item]

            frequency = data["frequency"]

            last_seen = data["last_seen"]

            item_probability = data["probability"]

            item_expected_value = data["expected_value"]

            size = data["size"]

            # Frequency
            frequency_factor = np.log1p(
                frequency
            )

            # Recency
            age = max(
                index - last_seen,
                1
            )

            recency_factor = 1.0 / (
                1.0 + np.log1p(age)
            )

            # Size penalty
            size_factor = 1.0 / (
                1.0 + np.log1p(size) / 10.0
            )

            # SmartEdge score
            score = (

                item_expected_value

                * item_probability

                * frequency_factor

                * recency_factor

                * size_factor
            )

            return score

        # ========================================================
        # FIND WORST OBJECT
        # ========================================================

        victim = min(
            cache,
            key=cache_score
        )

        victim_score = cache_score(
            victim
        )

        # ========================================================
        # SCORE NEW OBJECT
        # ========================================================

        new_frequency_factor = np.log1p(1)

        new_recency_factor = 1.0

        new_size_factor = 1.0 / (
            1.0 + np.log1p(new_size) / 10.0
        )

        new_score = (

            expected_value

            * probability

            * new_frequency_factor

            * new_recency_factor

            * new_size_factor
        )

        # ========================================================
        # REPLACE IF BETTER
        # ========================================================

        if new_score > victim_score:

            del cache[victim]

            evictions += 1

            cache[url] = {

                "frequency": 1,

                "last_seen": index,

                "probability": probability,

                "expected_value": expected_value,

                "size": new_size
            }

            admissions += 1

        else:

            rejected += 1

    # ============================================================
    # FINAL STATISTICS
    # ============================================================

    hit_rate = (
        hits / total_requests
    ) * 100 if total_requests else 0.0

    average_probability = (
        total_probability / total_requests
        if total_requests
        else 0.0
    )

    average_expected_value = (
        total_expected_value / total_requests
        if total_requests
        else 0.0
    )

    return {

        "hits": hits,

        "misses": misses,

        "hit_rate": hit_rate,

        "evictions": evictions,

        "admissions": admissions,

        "rejected": rejected,

        "cache_size": len(cache),

        "average_probability":
            average_probability,

        "average_expected_value":
            average_expected_value
    }

# ============================================================
# RUN EXPERIMENT
# ============================================================

print("\n" + "=" * 65)

print("SMARTEDGE REAL-DATA EXPERIMENT")

print("=" * 65)

print(
    f"Requests       : {REQUEST_LIMIT}"
)

print(
    f"Cache capacity : {CACHE_CAPACITY}"
)

print(
    f"Threshold      : {SMARTEDGE_THRESHOLD}"
)


# ============================================================
# LRU
# ============================================================

print("\nRunning LRU...")

lru = run_lru(
    requests,
    CACHE_CAPACITY
)


# ============================================================
# LFU
# ============================================================

print("Running LFU...")

lfu = run_lfu(
    requests,
    CACHE_CAPACITY
)


# ============================================================
# SMARTEDGE
# ============================================================

print("Running SmartEdge...")

smartedge = run_smartedge(
    requests,
    CACHE_CAPACITY,
    SMARTEDGE_THRESHOLD
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 65)

print("FINAL RESULTS")

print("=" * 65)

print(
    f"LRU       : "
    f"{lru['hit_ratio'] * 100:.2f}%"
)

print(
    f"LFU       : "
    f"{lfu['hit_ratio'] * 100:.2f}%"
)

print(
    f"SmartEdge : "
    f"{smartedge['hit_ratio'] * 100:.2f}%"
)

print("\nEvictions")

print(
    "LRU       :",
    lru["evictions"]
)

print(
    "LFU       :",
    lfu["evictions"]
)

print(
    "SmartEdge :",
    smartedge["evictions"]
)

print("\nSmartEdge ML")

print(
    "Predictions         :",
    REQUEST_LIMIT
)

print(
    "Cache admissions    :",
    smartedge["admissions"]
)

print(
    "Average probability :",
    f"{smartedge['average_probability']:.4f}"
)

print(
    "Cache size          :",
    smartedge["cache_size"]
)

print("=" * 65)